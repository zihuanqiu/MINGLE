import logging
import time
import torch
log = logging.getLogger(__name__)


class timeit_context:
    """
    Usage:

    ```python
    with timeit_context() as timer:
        ... # code block to be measured
    ```
    """

    nest_level = -1

    def _log(self, msg):
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        Sets the start time and logs an optional message indicating the start of the code block execution.

        Args:
            msg: str, optional message to log
        """
        self.start_time = time.time()
        timeit_context.nest_level += 1
        if self.msg is not None:
            self._log("  " * timeit_context.nest_level + "[BEGIN] " + str(self.msg))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calculates the elapsed time and logs it, along with an optional message indicating the end of the code block execution.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._log(
            "  " * timeit_context.nest_level
            + "[END]   "
            + str(f"Elapsed time: {elapsed_time:.2f}s")
        )
        timeit_context.nest_level -= 1


def get_gpu_memory_usage():
    """返回当前GPU内存的使用量，单位是MB"""
    return torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为 GB

class gpu_mem_context:
    """
    用于统计 GPU 内存的上下文管理器
    Usage:
    ```python
    with gpu_mem_context() as mem_tracker:
        ... # code block to be measured
    ```
    """

    nest_level = -1

    def _log(self, msg):
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        记录函数开始时的 GPU 内存使用量，并记录开始的日志信息
        """
        self.start_memory = get_gpu_memory_usage()  # 获取开始时的内存使用
        self.peak_memory = self.start_memory  # 初始化峰值内存
        gpu_mem_context.nest_level += 1
        if self.msg is not None:
            self._log("  " * gpu_mem_context.nest_level + "[BEGIN] " + str(self.msg))
        self._log("  " * gpu_mem_context.nest_level + f"Initial GPU memory usage: {self.start_memory:.2f} GB")
        return self  # Ensure the context manager returns itself for further use

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        计算并记录 GPU 内存的变化，输出结束时的内存使用信息和峰值内存使用量
        """
        end_memory = get_gpu_memory_usage()  # 获取结束时的内存使用
        memory_change = end_memory - self.start_memory  # 计算内存变化

        self._log(
            "  " * gpu_mem_context.nest_level
            + "[END]   "
            + str(f"GPU memory change: {memory_change:.2f} GB")
        )
        self._log(
            "  " * gpu_mem_context.nest_level
            + f"Final GPU memory usage: {end_memory:.2f} GB"
        )
        self._log(
            "  " * gpu_mem_context.nest_level
            + f"Peak GPU memory usage during execution: {self.peak_memory:.2f} GB"
        )
        gpu_mem_context.nest_level -= 1

    def update_peak_memory(self):
        """检查并更新峰值内存"""
        current_memory = get_gpu_memory_usage()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory