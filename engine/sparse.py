"""Inference-only sparse paths (no autodiff): CSR forward and dead-unit compaction."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSRMatrix:
    """Rows = output units, built from W.T (W is (in, out))."""
    data: np.ndarray     # (nnz,) values
    indices: np.ndarray  # (nnz,) int32 input indices
    indptr: np.ndarray   # (out+1,) int32
    shape: tuple[int, int]  # (out, in)


def csr_from_dense(W: np.ndarray, mask: np.ndarray) -> CSRMatrix:
    Wt, Mt = W.T, mask.T
    rows, cols = np.nonzero(Mt)  # row-major order: sorted by output unit
    indptr = np.zeros(Wt.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=Wt.shape[0]), out=indptr[1:])
    return CSRMatrix(Wt[rows, cols].copy(), cols.astype(np.int32), indptr, Wt.shape)


def sparse_linear(x: np.ndarray, W: CSRMatrix, b: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], W.shape[0]), dtype=np.result_type(x, W.data))
    starts = W.indptr[:-1]
    nonempty = W.indptr[1:] > starts
    if nonempty.any():
        # reduceat quirk: an empty segment returns the element at its start
        # instead of 0, so reduce only the non-empty rows.
        prod = x[:, W.indices] * W.data
        out[:, nonempty] = np.add.reduceat(prod, starts[nonempty], axis=1)
    return out + b


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def compact_mlp(layers: list[tuple[np.ndarray, np.ndarray]],
                act: Callable[[np.ndarray], np.ndarray] = _relu
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop hidden units with no incoming or no outgoing weights; returns smaller
    dense (W, b). Input W must already be masked (W * mask). A unit with no
    incoming weights outputs the constant act(b_j): it is folded into the next
    layer's bias so the function is unchanged exactly."""
    layers = [(W.copy(), b.copy()) for W, b in layers]
    changed = True
    while changed:
        changed = False
        for l in range(len(layers) - 1):
            W, b = layers[l]
            Wn, bn = layers[l + 1]
            no_in = ~W.any(axis=0)
            no_out = ~Wn.any(axis=1)
            fold = no_in & ~no_out
            if fold.any():
                bn = bn + act(b[fold]) @ Wn[fold]
            keep = ~(no_in | no_out)
            if not keep.all():
                changed = True
                layers[l] = (W[:, keep], b[keep])
                layers[l + 1] = (Wn[keep], bn)
            elif fold.any():
                layers[l + 1] = (Wn, bn)
    return layers


def count_flops(layers: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Per example: 2 * nnz (multiply + add) + out (bias) per layer."""
    return int(sum(2 * np.count_nonzero(W) + W.shape[1] for W, _ in layers))
