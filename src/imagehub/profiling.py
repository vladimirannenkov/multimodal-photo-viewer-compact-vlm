"""Process-local measurements of model calls; snapshots can be saved with evaluation runs."""

import functools
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager

_lock = threading.Lock()
_stats = {}
_session = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def memory():
    result = {"rss_bytes": None, "cuda_allocated_bytes": None, "cuda_reserved_bytes": None}
    try:
        import psutil

        result["rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        result["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        result["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    return result


def cpu_seconds():
    try:
        import psutil

        times = psutil.Process(os.getpid()).cpu_times()
        return times.user + times.system
    except (ImportError, OSError):
        return None


@contextmanager
def measure(name):
    before, start, cpu_before, failed = memory(), time.perf_counter(), cpu_seconds(), False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed, after, cpu_after = time.perf_counter() - start, memory(), cpu_seconds()
        with _lock:
            s = _stats.setdefault(
                name,
                {
                    "calls": 0,
                    "errors": 0,
                    "seconds": 0.0,
                    "max_observed_rss_bytes": 0,
                    "max_observed_cuda_allocated_bytes": 0,
                    "max_observed_cuda_reserved_bytes": 0,
                    "max_rss_increase_bytes": 0,
                    "max_cuda_allocated_increase_bytes": 0,
                    "cpu_seconds": 0.0,
                },
            )
            s["calls"] += 1
            s["errors"] += int(failed)
            s["seconds"] += elapsed
            if cpu_before is not None and cpu_after is not None:
                s["cpu_seconds"] += max(0.0, cpu_after - cpu_before)
            for source, target in [
                ("rss_bytes", "max_observed_rss_bytes"),
                ("cuda_allocated_bytes", "max_observed_cuda_allocated_bytes"),
                ("cuda_reserved_bytes", "max_observed_cuda_reserved_bytes"),
            ]:
                s[target] = max(s[target], before[source] or 0, after[source] or 0)
            for source, target in [
                ("rss_bytes", "max_rss_increase_bytes"),
                ("cuda_allocated_bytes", "max_cuda_allocated_increase_bytes"),
            ]:
                if before[source] is not None and after[source] is not None:
                    s[target] = max(s[target], after[source] - before[source])


def tracked(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{name}:{args[0]}" if name == "VLM.generate" and args else name
            with measure(key):
                result = fn(*args, **kwargs)
            if name == "VLM.generate" and args and str(args[0]).startswith("api:"):
                _observe_ollama(key, str(args[0])[4:])
            return result

        return wrapper

    return decorate


def _observe_ollama(key, model):
    """Record Ollama's reported model footprint; unavailable on other API servers."""
    import httpx

    from . import llm

    try:
        url = llm.effective_base_url().rstrip("/").removesuffix("/v1") + "/api/ps"
        response = httpx.get(url, timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        entry = next((m for m in models if m.get("name") in (model, f"{model}:latest")), None)
        if entry:
            with _lock:
                stats = _stats[key]
                for field in ("size", "size_vram"):
                    value = entry.get(field)
                    if isinstance(value, int):
                        stats[f"ollama_max_observed_{field}_bytes"] = max(
                            stats.get(f"ollama_max_observed_{field}_bytes", 0), value
                        )
    except Exception:  # Telemetry must never break model inference.
        pass


def snapshot():
    with _lock:
        return {k: dict(v) for k, v in _stats.items()}


def save_snapshot(kind, summary, *, per_event=False, baseline=None):
    """Persist metrics: a running process gets one file, individual requests get their own."""
    from .config import get_settings

    folder = get_settings().data_dir / "evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex if per_event else _session
    path = folder / f"{kind}_{suffix}.json"
    temporary = folder / f".{path.name}.{uuid.uuid4().hex}.tmp"
    values = snapshot()
    if baseline is not None:
        for key in list(values):
            prior = baseline.get(key, {})
            for field in ("calls", "errors", "seconds", "cpu_seconds"):
                if field in values[key]:
                    values[key][field] = max(0, values[key][field] - prior.get(field, 0))
            if not values[key]["calls"]:
                del values[key]
    payload = {
        "summary": {"kind": kind, "cumulative_profile": baseline is None, **summary},
        "profile": values,
        "memory": memory(),
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
