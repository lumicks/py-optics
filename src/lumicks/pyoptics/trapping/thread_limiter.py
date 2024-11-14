from contextlib import contextmanager

from numba import get_num_threads, set_num_threads
from numba.core.config import NUMBA_NUM_THREADS


@contextmanager
def thread_limiter(number_of_threads: int):
    level = get_num_threads()
    if number_of_threads < 1 or number_of_threads > NUMBA_NUM_THREADS:
        raise ValueError(
            f"Number of threads needs to be 1 or more, and less than {NUMBA_NUM_THREADS}"
        )
    if not isinstance(number_of_threads, int):
        raise RuntimeError("number_of_threads needs to be an integer")
    set_num_threads(number_of_threads)
    try:
        yield level
    finally:
        set_num_threads(level)
