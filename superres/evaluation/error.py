import cupy as cp
import numpy as np

from typing import Callable


def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        print(f"Warning: mismatched shapes {a.shape} and {b.shape}, "
              f"taking the smaller dimensions")
        min_slice = tuple(
            slice(0, min(a_dim, b_dim), None)
            for a_dim, b_dim in zip(a.shape, b.shape)
        )
        a = a[min_slice]
        b = b[min_slice]

    return float(cp.mean(
        cp.square(cp.array(b) - cp.array(a))))


ErrorFunction = Callable[[np.ndarray, np.ndarray], float]

ERROR_FUNCTIONS: dict[str, ErrorFunction] = {
    "mse": mse,
}


def named_error_function(name: str = "mse") -> ErrorFunction:
    if fn := ERROR_FUNCTIONS.get(name):
        return fn

    raise ValueError(
        f"Unknown error function {name}; "
        f"possible values are: {', '.join(ERROR_FUNCTIONS.keys())}",
    )
