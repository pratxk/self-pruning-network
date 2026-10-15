import numpy as np


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32) -> np.ndarray:
    """N(0, 2/fan_in): ReLU halves the second moment, the 2 restores it."""
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(dtype)


def glorot_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32) -> np.ndarray:
    """N(0, 2/(fan_in+fan_out)): small initial logits, loss ~ ln(10) at init."""
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / (fan_in + fan_out))).astype(dtype)


def zeros(n: int, dtype=np.float32) -> np.ndarray:
    return np.zeros(n, dtype=dtype)
