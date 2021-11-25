from contextlib import contextmanager
from typing import Generator


@contextmanager
def op_indicator(message: str) -> Generator[None, None, None]:
    print(message, end="... ", flush=True)
    try:
        yield None
        print("DONE")
    except Exception:
        print("FAILED")
        raise
