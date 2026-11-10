"""A private CPU-only Ollama instance, isolated from the user's normal model server."""

import atexit
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import httpx
import psutil

from .config import REPO, get_settings

_lock = threading.Lock()
_process = None
BASE_URL = "http://127.0.0.1:11435/v1"


def ensure() -> str:
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return BASE_URL
        if sys.platform not in ("win32", "linux"):
            raise RuntimeError("Режим CPU для сервера моделей на этой платформе не поддерживается")
        try:
            with socket.create_connection(("127.0.0.1", 11435), timeout=0.5):
                raise RuntimeError("Порт 11435 занят: невозможно подтвердить, что сервер работает только на CPU")
        except (ConnectionRefusedError, TimeoutError, OSError):
            pass
        executable = shutil.which("ollama")
        local = REPO / "data" / "ollama" / ("ollama.exe" if sys.platform == "win32" else "ollama")
        if local.is_file():
            executable = str(local)
        if not executable:
            raise RuntimeError("Для режима CPU требуется локальная установка Ollama")
        env = os.environ.copy()
        env.update(
            {
                "OLLAMA_HOST": "127.0.0.1:11435",
                "CUDA_VISIBLE_DEVICES": "-1",
                "HIP_VISIBLE_DEVICES": "-1",
                "ROCR_VISIBLE_DEVICES": "-1",
                "GGML_VK_VISIBLE_DEVICES": "-1",
                "OLLAMA_VULKAN": "0",
                "OLLAMA_NO_CLOUD": "1",
                "OLLAMA_MAX_LOADED_MODELS": "1",
            }
        )
        log_path = get_settings().data_dir / "ollama_cpu.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            _process = subprocess.Popen(
                [executable, "serve"],
                env=env,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
            )
        for _ in range(100):
            if _process.poll() is not None:
                raise RuntimeError(f"CPU-сервер Ollama завершился с кодом {_process.returncode}; см. {log_path}")
            try:
                response = httpx.get("http://127.0.0.1:11435/api/version", timeout=0.5, trust_env=False)
                listeners = psutil.Process(_process.pid).net_connections(kind="inet")
                owns_port = any(c.status == psutil.CONN_LISTEN and c.laddr.port == 11435 for c in listeners)
                if response.status_code == 200 and owns_port:
                    return BASE_URL
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        stop_unlocked()
        raise RuntimeError(f"CPU-сервер Ollama не запустился; см. {log_path}")


def stop_unlocked() -> None:
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
    _process = None


def stop() -> None:
    with _lock:
        stop_unlocked()


atexit.register(stop)
