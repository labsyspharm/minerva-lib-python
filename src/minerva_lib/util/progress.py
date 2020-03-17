import os, sys, threading

class ProgressPercentage:

    def __init__(self):
        self._total_size = 0
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def add(self, filename):
        self._total_size += float(os.path.getsize(filename))

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._total_size) * 100
            sys.stdout.write(
                "\r%s MB / %s MB (%.1f%%)" % (
                    self._seen_so_far // 1000000, self._total_size // 1000000,
                    percentage))
            sys.stdout.flush()