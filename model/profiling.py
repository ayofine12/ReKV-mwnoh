import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch


class _TimerToken:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        if use_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()
            self.end = None

    def stop(self):
        if self.use_cuda:
            self.end.record()
            self.end.synchronize()
            return self.start.elapsed_time(self.end)
        return (time.perf_counter() - self.start) * 1000.0


class ReKVProfiler:
    def __init__(self):
        self.enabled = False
        self.output_path = None
        self._stats = defaultdict(lambda: {
            "count": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": None,
        })
        self._metadata = {}
        self._lock = threading.Lock()
        self._tls = threading.local()

    def configure(self, enabled=False, output_path=None, reset=False):
        with self._lock:
            self.enabled = enabled
            if output_path is not None:
                self.output_path = output_path
            if reset:
                self._stats.clear()
                self._metadata = {}

    def update_metadata(self, **kwargs):
        with self._lock:
            self._metadata.update(kwargs)

    def is_enabled(self):
        return self.enabled

    def _phase_stack(self):
        stack = getattr(self._tls, "phase_stack", None)
        if stack is None:
            stack = []
            self._tls.phase_stack = stack
        return stack

    def _qualified_name(self, name):
        stack = self._phase_stack()
        if not stack:
            return name
        return "::".join(stack + [name])

    def start(self, name):
        if not self.enabled:
            return None
        return self._qualified_name(name), _TimerToken(torch.cuda.is_available())

    def stop(self, token):
        if token is None:
            return
        name, timer = token
        elapsed_ms = timer.stop()
        with self._lock:
            stat = self._stats[name]
            stat["count"] += 1
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["min_ms"] = elapsed_ms if stat["min_ms"] is None else min(stat["min_ms"], elapsed_ms)

    @contextmanager
    def section(self, name):
        token = self.start(name)
        record_ctx = torch.autograd.profiler.record_function(name) if self.enabled else nullcontext()
        with record_ctx:
            try:
                yield
            finally:
                self.stop(token)

    @contextmanager
    def phase(self, name):
        stack = self._phase_stack()
        stack.append(name)
        try:
            yield
        finally:
            stack.pop()

    def make_module_hooks(self, name):
        attr_name = f"_rekv_profile_tokens_{name.replace('.', '_')}"

        def pre_hook(module, _inputs):
            if not self.enabled:
                return
            tokens = getattr(module, attr_name, None)
            if tokens is None:
                tokens = []
                setattr(module, attr_name, tokens)
            tokens.append(self.start(name))

        def post_hook(module, _inputs, _output):
            if not self.enabled:
                return
            tokens = getattr(module, attr_name, None)
            if not tokens:
                return
            self.stop(tokens.pop())

        return pre_hook, post_hook

    def summary(self):
        with self._lock:
            metrics = {}
            for name, stat in sorted(self._stats.items()):
                avg_ms = stat["total_ms"] / stat["count"] if stat["count"] else 0.0
                metrics[name] = {
                    "count": stat["count"],
                    "total_ms": round(stat["total_ms"], 6),
                    "avg_ms": round(avg_ms, 6),
                    "max_ms": round(stat["max_ms"], 6),
                    "min_ms": round(stat["min_ms"] or 0.0, 6),
                }
            return {
                "enabled": self.enabled,
                "metadata": dict(self._metadata),
                "metrics": metrics,
            }

    def dump(self, output_path=None):
        if not self.enabled:
            return
        path = output_path or self.output_path
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=True)


_GLOBAL_PROFILER = ReKVProfiler()


def get_profiler():
    return _GLOBAL_PROFILER


def configure_profiling(enabled=False, output_path=None, reset=False):
    _GLOBAL_PROFILER.configure(enabled=enabled, output_path=output_path, reset=reset)
    return _GLOBAL_PROFILER


@contextmanager
def profile_section(name):
    with _GLOBAL_PROFILER.section(name):
        yield


@contextmanager
def profile_phase(name):
    with _GLOBAL_PROFILER.phase(name):
        yield
