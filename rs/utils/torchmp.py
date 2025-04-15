import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
import torch.multiprocessing as mp
import time
import signal
import sys
import tensorflow as tf
import numpy
from typing import Callable, Tuple


class TorchLeakExecutor:
    """
    CUDA-safe memory-leak containment using torch.multiprocessing
    with forkserver optimization for Linux systems
    """

    def __init__(self, device_id=0, timeout=60):
        """
        :param device_id: CUDA device to isolate (default: 0)
        :param timeout: Maximum execution time in seconds (default: 60)
        """
        self.device_id = device_id
        self.timeout = timeout
        self.process = None
        self.result_queue = mp.Queue()  # Queue for retrieving results
        self._configure_environment()

    def _configure_environment(self):
        """Set up forkserver and CUDA parameters"""
        mp.set_start_method("forkserver", force=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        mp.set_forkserver_preload(["torch"])
        torch.multiprocessing.set_sharing_strategy("file_system")

    def execute(self, target, args=()):
        """
        Execute leaky CUDA function in isolated process

        :param target: Callable to execute
        :param args: Arguments for target function
        """
        ctx = mp.get_context("forkserver")
        self.process = ctx.Process(target=self._cuda_worker, args=(target, args), daemon=True)
        self.process.start()
        self._monitor_process()

    def _cuda_worker(self, target, args):
        """Isolated CUDA process with forced cleanup"""
        # Initialize CUDA context
        torch.cuda.init()
        device = torch.device(f"cuda:{self.device_id}")

        try:
            with torch.cuda.device(device):
                result = target(*args)
                del result  # Explicit cleanup
        finally:
            # Finalize CUDA context
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _monitor_process(self):
        """Enforce resource limits and timeouts"""
        start_time = time.time()

        while self.process.is_alive():
            # Timeout check
            if time.time() - start_time > self.timeout:
                self._safe_terminate()
                raise TimeoutError(f"Process exceeded {self.timeout}s limit")

            # GPU memory check (requires nvidia-smi)
            try:
                mem_used = self._get_gpu_memory()
                print(f"GPU memory used: {mem_used}MB")
                if mem_used > 5100:  # 5GB limit
                    self._safe_terminate()
                    raise MemoryError(f"GPU memory exceeded 1GB (used: {mem_used}MB)")
            except RuntimeError:
                pass

            time.sleep(0.1)

    def _get_gpu_memory(self):
        """Get current process GPU memory usage"""
        pid = self.process.pid
        result = os.popen(
            f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        ).read()
        for line in result.split("\n"):
            if str(pid) in line:
                return float(line.split(", ")[1])
        return 0

    def _safe_terminate(self):
        """Guaranteed process cleanup sequence"""
        if self.process.is_alive():
            os.kill(self.process.pid, signal.SIGKILL)
            self.process.join()
            self.process.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._safe_terminate()


def leaky_simulation():
    """CUDA simulation with intentional memory leak"""
    cache = []
    for _ in range(100):
        # Allocate and retain 10MB tensor
        tensor = torch.randn(1280, 1280, device="cuda")  # ~10MB
        cache.append(tensor)
        time.sleep(0.01)
    return sum(t.mean() for t in cache)


# Usage Example
if __name__ == "__main__":

    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")

    with TorchLeakExecutor(device_id=0, timeout=10) as executor:
        executor.execute(leaky_simulation)

    print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")
