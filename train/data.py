"""Fashion-MNIST loader. scikit-learn is used ONLY here, only to download the
dataset (fetch_openml); nothing else in the repo may import it."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHA256 = "75979490032625d6583ff7101de6f9243d07d0a8d33d3c9b3b7e79b4c59382af"  # uint8 X + int64 y
SPLIT_SEED = 1234  # fixed across all runs, independent of the run seed
N_TRAIN, N_VAL = 50_000, 10_000


@dataclass(frozen=True)
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _checksum(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


ZALANDO = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def _download_zalando() -> tuple[np.ndarray, np.ndarray]:
    """Fallback: the original idx files. Same 70k images in the same order as OpenML
    (60k train then 10k test); load_raw's SHA-256 check confirms it."""
    import gzip
    import urllib.request

    def read(name: str, offset: int) -> np.ndarray:
        with urllib.request.urlopen(ZALANDO + name, timeout=120) as r:
            return np.frombuffer(gzip.decompress(r.read()), np.uint8, offset=offset)

    X = np.concatenate([read("train-images-idx3-ubyte.gz", 16),
                        read("t10k-images-idx3-ubyte.gz", 16)]).reshape(-1, 784)
    y = np.concatenate([read("train-labels-idx1-ubyte.gz", 8),
                        read("t10k-labels-idx1-ubyte.gz", 8)])
    return X, y


def _download(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    try:
        X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True,
                            data_home=str(cache.parent / "openml"), parser="liac-arff")
    except Exception as openml_err:  # network / openml outage
        print(f"OpenML download failed ({type(openml_err).__name__}); trying the Zalando mirror")
        try:
            X, y = _download_zalando()
        except Exception as e:
            raise RuntimeError(
                "Could not download Fashion-MNIST from OpenML or the Zalando mirror. "
                "Download it manually from https://www.openml.org/d/40996 or retry later.") from e
    X, y = X.astype(np.uint8), y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_raw(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    cache = Path(data_dir) / "fashion_mnist.npz"
    if cache.exists():
        with np.load(cache) as f:
            X, y = f["X"], f["y"]
    else:
        X, y = _download(cache)
    if _checksum(X, y) != SHA256:
        raise RuntimeError(f"checksum mismatch for {cache}: delete it and re-run")
    return X, y


def load_fashion_mnist(data_dir: str = "data", dtype=np.float32) -> Splits:
    """60k official train -> 50k train + 10k val (seeded split); 10k official test.
    Pixels / 255, then standardized with the 50k-train mean / std (global scalars)."""
    X, y = load_raw(data_dir)
    X = X.astype(np.float64) / 255.0
    perm = np.random.default_rng(SPLIT_SEED).permutation(60_000)
    tr, va = perm[:N_TRAIN], perm[N_TRAIN:]
    mean, std = X[tr].mean(), X[tr].std()
    Xs = ((X - mean) / std).astype(dtype)
    return Splits(Xs[tr], y[tr], Xs[va], y[va], Xs[60_000:], y[60_000:])
