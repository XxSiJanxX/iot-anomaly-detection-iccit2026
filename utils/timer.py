"""Simple context-manager timer for recording training / inference time."""
import time

class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.4f}s")

    def __str__(self):
        return f"{self.elapsed:.4f}s"
