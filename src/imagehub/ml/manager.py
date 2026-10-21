"""Один замок на GPU: фоновый пайплайн и запросы пользователя (чат) не загружают модели одновременно."""

import sys
import threading
import time
from contextlib import contextmanager

gpu_lock = threading.RLock()
_user_waiting = 0
_cnt_lock = threading.Lock()


@contextmanager
def gpu(user=False):
    """user=True: интерактивный запрос. Пайплайн проверяет user_waiting() между пакетами и уступает."""
    global _user_waiting
    if user:
        with _cnt_lock:
            _user_waiting += 1
    try:
        while not user and _user_waiting:  # фоновая обработка уступает очередь пользователю
            time.sleep(0.05)
        with gpu_lock:
            yield
    finally:
        if user:
            with _cnt_lock:
                _user_waiting -= 1


def user_waiting() -> bool:
    return _user_waiting > 0


_resident = None


def ensure(group):
    """В видеопамяти держим одну группу тяжёлых моделей (ocr, fix, scene, vqa). SigLIP2 мал и остаётся."""
    global _resident
    if _resident != group:
        free_heavy()
        _resident = group


def trim_memory():
    """Возвращает освобождённую память системе (на Windows рабочий набор процесса не сжимается сам)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        import os

        if os.name == "nt":
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(k.GetCurrentProcess()))
    except Exception:
        pass


def free_heavy():
    global _resident
    from . import textfix, vlm

    vlm.free()
    textfix.free()
    florence = sys.modules.get("imagehub.ml.florence")  # импортируется (и тянет torch) только если сцена уже обрабатывалась
    if florence is not None:
        florence.free()
    _resident = None
    trim_memory()
