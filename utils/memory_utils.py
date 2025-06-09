import psutil


class MemoryGuard:
    def __init__(self, max_memory_usage=0.9):
        self.max_memory_usage = max_memory_usage

    def __enter__(self):
        if self._system_memory_usage() > self.max_memory_usage:
            raise MemoryError("System memory threshold exceeded")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _system_memory_usage(self):
        return psutil.virtual_memory().percent / 100


# # Usage in your methods:
# def _perform_ocr_sync(self, img: np.ndarray) -> str:
#     with MemoryGuard(max_memory_usage=0.85):  # 85% system memory
#         # Your existing OCR code
#         pass