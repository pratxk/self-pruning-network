import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """The single seeded Generator passed everywhere (init, shuffling, tie-breaks)."""
    return np.random.default_rng(seed)
