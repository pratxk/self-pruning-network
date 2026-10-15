import numpy as np

from engine.sparse import compact_mlp, count_flops, csr_from_dense, sparse_linear


def _masked_forward(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def test_csr_round_trip_with_empty_rows():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5))
    mask = rng.random((6, 5)) < 0.4
    mask[:, [0, 4]] = False  # empty first and last output rows (reduceat quirk)
    x = rng.standard_normal((3, 6))
    out = sparse_linear(x, csr_from_dense(W, mask), np.ones(5))
    assert np.allclose(out, x @ (W * mask) + 1)
    assert np.allclose(out[:, [0, 4]], 1.0)


def test_csr_all_empty():
    out = sparse_linear(np.ones((2, 3)), csr_from_dense(np.ones((3, 4)), np.zeros((3, 4), bool)),
                        np.arange(4.0))
    assert np.array_equal(out, np.tile(np.arange(4.0), (2, 1)))


def test_compaction_preserves_outputs_and_shrinks():
    rng = np.random.default_rng(1)
    W1, W2, W3 = (rng.standard_normal(s) for s in [(8, 6), (6, 5), (5, 3)])
    b1, b2, b3 = (rng.standard_normal(n) for n in (6, 5, 3))
    W1[:, 0] = 0          # unit 0 of hidden 1: no incoming -> constant relu(b) folded
    W2[1, :] = 0          # unit 1 of hidden 1: no outgoing -> dropped
    W3[2, :] = 0          # unit 2 of hidden 2: no outgoing
    layers = [(W1, b1), (W2, b2), (W3, b3)]
    x = rng.standard_normal((4, 8))
    compact = compact_mlp(layers)
    assert np.allclose(_masked_forward(compact, x), _masked_forward(layers, x))
    assert compact[0][0].shape == (8, 4) and compact[1][0].shape == (4, 4)


def test_flop_count_matches_hand_count():
    W1 = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])  # nnz 3, out 2
    W2 = np.array([[1.0], [0.0]])                         # nnz 1, out 1
    assert count_flops([(W1, np.zeros(2)), (W2, np.zeros(1))]) == (2 * 3 + 2) + (2 * 1 + 1)
