# DESIGN

Four questions, in order. Each section opens with the answer; derivations follow.
Every number here traces to a committed file under `results/` or to a test.

## 1. Deriving the importance criterion

**Answer.** A weight's importance is the loss increase expected from setting it
to zero. Optimal Brain Damage with the diagonal Fisher as the curvature gives
`s_i = ½ v̂_i w_i²`, where `v̂` is Adam's bias-corrected second moment
(`prune/criteria.py::FisherOBD`). Weights with the lowest `s_i` are pruned,
ranked globally across layers. The full second-order Taylor form
`½ B v̂ w² − m̂ w` is kept as the ablation `fisher_obd_fo`. Including the
first-order term measurably hurts; the reasons are below.

**Derivation.** Removing weight *i* moves it by `δ = −w_i e_i`. Expanding the
loss around the current weights:

```
ΔL_i = gᵀδ + ½ δᵀ H δ = −g_i w_i + ½ H_ii w_i²
```

- **The Hessian, cheaply.** For cross-entropy (a negative log-likelihood) the
  Fisher information equals the Gauss-Newton matrix. It approximates the Hessian
  near a minimum where residuals are small. The diagonal empirical Fisher is
  `E_n[g_{n,i}²]` over per-example gradients. Adam's `v̂` is an EMA of squared
  gradients, so it gives the curvature at no extra cost: no second backward pass.
- **Second-order term.** Near convergence `E[g] ≈ 0`, so `½ H_ii w_i²`
  dominates. That is Optimal Brain Damage (LeCun et al., 1990), and it is the
  primary criterion.
- **Why the first-order term is dropped.** Pruning ramps from epoch 3 to 20,
  while training is still moving, so `m̂` is far from 0. The term `−m̂_i w_i`
  linearly extrapolates the loss over a step of the full size `|w_i|`. That is
  not a small step. The term is most negative exactly where Adam is already
  shrinking a large weight, so it ranks those weights for removal first.
  Measured: `fisher_obd_fo` vs `fisher_obd` in `results/part4*/sweep.csv`.

**Caveats.**
1. **`v̂` is a mini-batch Fisher, biased low by ≈B.** Adam's `v` averages
   `ḡ²`, the square of the batch-*mean* gradient:
   `E[ḡ²] = E[g]² + Var(g_n)/B`. Near a minimum the noise term dominates, so
   per-example Fisher ≈ `B·v̂` (128× larger here). For `½ v̂ w²` alone this is
   a uniform scale and does not change the ranking. It matters when mixing with
   the first-order term: without `B`, the curvature term is 128× too small and
   `−m̂w` dominates. `fisher_obd_fo` therefore uses `½ B v̂ w² − m̂ w`.
2. The diagonal ignores interactions between weights removed together. That is
   why pruning is gradual (cubic schedule, `prune/schedule.py`) with training
   in between. The one-shot ablation (`magnitude_oneshot`) measures the cost of
   ignoring this.
3. The expansion is local: valid for small `δ`. A large weight's removal is not
   small, which is another reason to prune gradually.

**Why it should rank across layers better than magnitude.** Magnitude assumes
every weight has the same curvature. Layers are initialised at different scales
(He: `std = √(2/fan_in)`), so raw `|w|` is not comparable across layers. That
forces magnitude pruning into per-layer quotas (our `magnitude_layer`
ablation). `s_i` is in units of loss, so a global threshold is meaningful.
Example: a large weight on a border pixel of Fashion-MNIST that is almost
always 0 has tiny `v̂` and costs nothing to remove.

References: LeCun et al. 1990 (OBD); Molchanov et al. 2017/2019 (Taylor
importance); Zhu & Gupta 2017 (cubic schedule); Evci et al. 2020 (RigL).

## 2. The gradient of a masked weight

**Answer.** There are two different, well-defined quantities, and the engine
exposes both:

| Quantity | At a pruned position | Meaning | Used by |
| --- | --- | --- | --- |
| `∂L/∂W = M ⊙ ∂L/∂W_eff` (`W.grad`) | exactly 0 | the parameter does not affect the loss | the optimizer |
| `∂L/∂W_eff` (`MaskedLinear.dense_grad()`) | generally ≠ 0 | how the loss would change if the connection existed | regrowth scoring |

`MaskedLinear.forward` builds `W_eff = W * M` as a graph node, then
`y = x @ W_eff + b`. The `mul` backward produces `W.grad = W_eff.grad * M`, which
is exactly 0.0 where `M = 0`. The layer keeps a reference to `W_eff`, so its
gradient (the counterfactual) survives `backward()`.
Tests: `test_param_grad_exactly_zero_at_masked_positions` and
`test_dense_grad_is_the_counterfactual` (finite differences with the mask
lifted).

**The bug most implementations have: optimizer drift.** A zero gradient is not
a zero update. A weight pruned at step *t* still has non-zero Adam moment `m`.
The update `m̂ / (√v̂ + ε)` keeps moving it away from 0 while `m` decays. The
"pruned" weight silently comes back while the mask says it is dead.
`test_naive_adam_drifts` and `test_naive_sgd_momentum_drifts` (xfail-strict)
prove the naive optimizer fails. `test_adam_no_drift` proves the fix.

**Fix (`nn/optim.py`).**
1. On prune: `W = m = v = t = 0` at those indices. Stale moments describe a
   weight that no longer exists.
2. Every step: `W *= M` after the update (hard-zero invariant).
3. On revival: `W = 0` (the function is unchanged at that instant),
   `m = v = 0`, and the element's step counter restarts.

All mask edits go through one choke point, `prune.pruner.apply_mask`, which
computes the pruned and revived sets and calls `optimizer.on_mask_change`.

**Why rule 3 needs a per-element step counter.** Standard Adam uses one global
`t` for bias correction. Revive a weight at global step `t` with `m = v = 0`.
Its first update is

```
Δw = −η · [(1−β₁) g / (1−β₁ᵗ)] / √[(1−β₂) g² / (1−β₂ᵗ)]
```

A fresh parameter (`t = 1`) gets exactly `η`. With a large global `t` the
factor tends to `0.1/√0.001 ≈ 3.16`. It is **2.515× at t = 1001**, because
`β₂^1001 ≈ 0.37` has not yet vanished (`test_naive_adam_revival_first_step_too_big`
checks the exact formula). Storing `t` per element, incremented only where
`M = 1`, makes a revived weight follow exactly the trajectory of a brand-new
Adam parameter. `test_adam_revival_matches_fresh_adam_bit_for_bit` checks this
bit for bit over 50 steps. `test_mask_event_isolation` proves the reset does not
touch any other entry.

## 3. Where the engine bottlenecks and how to optimize it

**Answer.** Measured, not guessed: the optimizer, not the matmuls, was the
bottleneck. `cProfile` over one dense epoch (391 steps,
`uv run python -m train.run_profile`) put **80%** of the time in
`Adam._update`. The cause was the per-element bias correction `np.power(β, t)`,
which is needed because `t` is per element (section 2). It ran a scalar pow per
weight per step. A table of `1 − βᵗ` indexed by integer `t` turns it into a
gather.

| | Before | After | Source |
| --- | --- | --- | --- |
| One epoch, profiled | 2.124 s | 0.853 s (2.5×) | `results/profile/profile_{before,after}.txt` |
| One Adam step, LeNet-300-100 | 4.385 ms | 1.060 ms (4.1×) | same |
| `Adam._update` share of epoch | 80% | 52% (incl. `_bias` gather) | same |

The revival bit-for-bit test still passes after the change: both trajectories
read the same table.

| Bottleneck | Why | Optimization |
| --- | --- | --- |
| Per-element bias correction (fixed) | `np.power(float, int array)` runs a scalar pow per element | Table of `1−βᵗ` indexed by `t`: a gather instead of a transcendental. Next step: since `t` is mostly uniform, skip the gather when `t.min() == t.max()` |
| Matmul FLOPs (784×300 layer ≈ 88% of FLOPs) | Real compute | Already BLAS, float32; at high sparsity use CSR or compacted layers in the backward too |
| Python overhead per op | Each op allocates a Tensor and a closure; topological sort rebuilt every step | Fused `linear(x, W, M, b)` with one hand-written backward; cache the topo order for a static graph |
| Temporary allocations | Every op returns a new array; grads allocated fresh | Preallocated grad buffers, `out=` arguments, in-place accumulation |
| Mask multiply every forward | Extra 784×300 elementwise op | Rely on the hard-zero invariant in a fused op; keep the multiply only in debug mode |
| Fancy indexing in the CSR path | `x[:, indices]` materializes B × nnz | Block-CSR, or a C/Numba kernel in production |
| Single-threaded | Threads pinned for reproducibility | Release threads for throughput runs; data-parallel minibatches |

## 4. Serving a self-pruned model in a multi-tenant service at scale

**Answer.** Do not serve masks. Ship a frozen, compacted artifact per accuracy
and cost tier. Get speed from structure, batching and quantization, not from
unstructured zeros.

- **Unstructured sparsity rarely speeds up GPUs or vectorized CPUs.**
  `results/part4/bench.csv` shows where CSR crosses over dense BLAS on this
  machine. Ship the compacted model (dead units removed: real, smaller dense
  matrices). Or retrain with structured sparsity (neuron or block, or 2:4 for
  sparse tensor cores) using the same saliency summed per group.
- **Export a frozen artifact:** weights plus CSR or compacted format, version,
  and accuracy report. Never apply masks at request time: per-request masks
  break batching.
- **Throughput:** dynamic batching with a latency cap (e.g. 5 ms). Weights are
  loaded once and shared read-only across worker processes, with threads pinned
  per worker. Layer int8 quantization on top: sparsity and quantization compound.
- **Multi-tenant:** one shared base model per tier. Tenants needing a different
  accuracy/cost trade-off are routed by tenant config to pre-built sparsity
  tiers. Per-tenant quotas and isolation stop one tenant's burst from starving
  the others.
- **Safety:** shadow traffic against the dense model, per-tenant canary,
  monitoring of prediction agreement, accuracy proxies and p50/p99 latency, and
  instant rollback to dense.
