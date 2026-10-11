"""Run all MLX work on one dedicated thread.

MLX ties its streams to the thread that created them. meetre caches models
(Parakeet, the chat LLM) across recordings and chat turns, each of which runs
on a fresh worker thread — so a model built in one thread could be used from
another, which fails with "There is no Stream(cpu, 1) in current thread".
Funnelling every MLX call through a single long-lived thread makes the
thread that built a model always the one that uses it. (It also stops two
models from fighting over the GPU at once.)
"""

from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meetre-mlx")
_local = threading.local()


def _mark_and_call(fn, args, kwargs):
    _local.on_mlx = True
    return fn(*args, **kwargs)


def on_mlx_thread(fn):
    """Decorator: run ``fn`` on the MLX thread and wait for its result.

    Nested calls (an MLX function calling another) run inline, so they can't
    deadlock waiting on their own thread.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(_local, "on_mlx", False):
            return fn(*args, **kwargs)
        return _executor.submit(_mark_and_call, fn, args, kwargs).result()

    return wrapper
