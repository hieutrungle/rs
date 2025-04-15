import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import multiprocessing as mp
import time
import signal
import tensorflow as tf
from multiprocessing import Process, Queue
import multiprocessing


class TFLeakExecutor:
    """
    CUDA-safe memory-leak containment using multiprocessing
    for TensorFlow operations.
    """

    def __init__(self, device_id=0, timeout=60):
        """
        :param device_id: CUDA device to isolate (default: 0)
        :param timeout: Maximum execution time in seconds (default: 60)
        """
        self.device_id = device_id
        self.timeout = timeout
        self.process = None
        self._configure_environment()

    def _configure_environment(self):
        """Set up CUDA environment variables"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # Suppress TensorFlow warnings
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid memory fragmentation
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        mp.set_start_method("forkserver", force=True)  # Use forkserver for better isolation
        mp.set_forkserver_preload(["tensorflow", "sionna"])  # Preload TensorFlow for forkserver

    def execute(self, target, args=()):
        """
        Execute leaky TensorFlow function in isolated process.

        :param target: Callable to execute
        :param args: Arguments for target function
        :return: The return value of the target function
        """
        queue = mp.Queue()
        self.process = mp.Process(target=self._tf_worker, args=(target, args, queue))
        self.process.start()

        try:
            # Monitor the process for timeout or memory issues
            self._monitor_process()
        finally:
            # Ensure the process is joined to clean up resources
            if self.process.is_alive():
                self.process.join()

        # Retrieve the result from the queue
        if not queue.empty():
            result = queue.get()
            if isinstance(result, Exception):
                raise result  # Re-raise exception from the worker process
            return result
        else:
            raise RuntimeError("No result returned from the isolated process.")

    def _tf_worker(self, target, args, queue):
        """Isolated TensorFlow process with forced cleanup"""
        try:
            # Initialize TensorFlow GPU context
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)

            # Execute the target function
            result = target(*args)
            queue.put(result)  # Send result back to the main process
        except Exception as e:
            queue.put(e)  # Send exception back to the main process
        finally:
            # Explicit cleanup of TensorFlow resources
            tf.keras.backend.clear_session()

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
                if mem_used > 5100:  # 5GB limit
                    self._safe_terminate()
                    raise MemoryError(f"GPU memory exceeded 5GB (used: {mem_used}MB)")
            except RuntimeError:
                pass

            time.sleep(0.2)

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

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._safe_terminate()


def leaky_simulation():
    """TensorFlow simulation with intentional memory leak"""
    cache = []
    for _ in range(100):
        # Allocate and retain a 10MB tensor
        tensor = tf.random.normal((1280, 1280))  # ~10MB
        cache.append(tensor)
        time.sleep(0.01)
    val = sum(tf.reduce_mean(tensor).numpy() for tensor in cache)
    return val


# # Usage Example
if __name__ == "__main__":
    print(
        f"Initial GPU Memory: {tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6:.2f} MB"
    )

    with TFLeakExecutor(device_id=0, timeout=10) as executor:

        try:
            result = executor.execute(leaky_simulation)
            print(f"Result from isolated process: {result}")
        except Exception as e:
            print(f"Error occurred: {e}")

    print(
        f"Final GPU Memory: {tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6:.2f} MB"
    )


# class Multiprocessor:

#     def __init__(self, device_id=0):
#         self.processes = []
#         self.queue = Queue()
#         self.device_id = device_id
#         self._configure_environment()

#     def _configure_environment(self):
#         """Set up CUDA environment variables"""
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
#         os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
#         os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Avoid memory fragmentation
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#         mp.set_start_method("forkserver", force=True)  # Use forkserver for better isolation

#     @staticmethod
#     def _wrapper(func, queue, args, kwargs):
#         ret = func(*args, **kwargs)
#         queue.put(ret)

#     def run(self, func, *args, **kwargs):
#         ctx = multiprocessing.get_context("forkserver")
#         args2 = [func, self.queue, args, kwargs]
#         p = ctx.Process(target=self._wrapper, args=args2)
#         self.processes.append(p)
#         p.start()

#     def wait(self):
#         rets = []
#         for p in self.processes:
#             ret = self.queue.get()
#             rets.append(ret)
#         for p in self.processes:
#             p.join()
#         return rets
# import copy


# def leaky_simulation():
#     """TensorFlow simulation with intentional memory leak"""
#     cache = []
#     for _ in range(100):
#         # Allocate and retain a 10MB tensor
#         tensor = tf.random.normal((1280, 1280))  # ~10MB
#         cache.append(tensor)
#         time.sleep(0.01)
#     print(f"Cache size: {len(cache)}")
#     val = sum(tf.reduce_mean(tensor).numpy() for tensor in cache)
#     return val


# def _wrapper(func, queue, args, kwargs):
#     """Wrapper function to execute target function and put result in queue"""
#     ret = func(*args, **kwargs)
#     queue.put(copy.deepcopy(ret))


# class Multiprocessor:
#     def __init__(self, device_id=0):
#         self.processes = []
#         self.queue = Queue()
#         self.device_id = device_id
#         self._configure_environment()

#     def _configure_environment(self):
#         """Set up CUDA environment variables"""
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
#         os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#         mp.set_start_method("forkserver", force=True)

#     def run(self, func, *args, **kwargs):
#         """Run the target function in a separate process"""
#         ctx = mp.get_context("forkserver")
#         args2 = (func, self.queue, args, kwargs)
#         p = ctx.Process(target=_wrapper, args=args2)
#         self.processes.append(p)
#         p.start()
#         print(f"Process {p.pid} started.")
#         time.sleep(0.1)

#     def wait(self):
#         rets = []
#         for p in self.processes:
#             p.join()

#         for p in self.processes:
#             if not p.is_alive():
#                 print(f"Process {p.pid} finished.")
#             else:
#                 print(f"Process {p.pid} is still running.")
#             ret = self.queue.get(block=False)
#             # rets.append(ret)
#         return rets


# def run():
#     mp.set_start_method("spawn", force=True)
#     queue = mp.Queue()
#     processes = []
#     for _ in range(2):
#         process = mp.Process(target=_wrapper, args=(leaky_simulation, queue, (), {}))
#         processes.append(process)
#         process.start()

#     for process in processes:
#         process.join()

#     results = [queue.get() for _ in range(len(processes))]


# if __name__ == "__main__":
#     mpp = Multiprocessor()
#     mpp.run(leaky_simulation)
#     rets = mpp.wait()
#     print(rets)

#     # run()
