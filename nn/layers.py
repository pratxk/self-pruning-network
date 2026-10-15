import numpy as np

from engine import ops
from engine.tensor import Tensor
from nn import init
from nn.module import Module


class MaskedLinear(Module):
    """y = x @ (W * mask) + b. Every linear layer is one of these, so dense and
    pruned training share one code path (dense = all-ones mask).

    mask is a plain bool array, never a Tensor, never gets a gradient. Edit it
    only through prune.pruner.apply_mask (in place), never directly."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator,
                 init_fn=init.he_normal, dtype=np.float32):
        self.W = Tensor(init_fn(fan_in, fan_out, rng, dtype), requires_grad=True)
        self.b = Tensor(init.zeros(fan_out, dtype), requires_grad=True)
        self.mask = np.ones((fan_in, fan_out), dtype=bool)
        self._w_eff: Tensor | None = None  # kept so its grad survives backward

    def forward(self, x):
        self._w_eff = self.W * Tensor(self.mask, dtype=self.W.data.dtype)
        return x @ self._w_eff + self.b

    def dense_grad(self) -> np.ndarray:
        """dL/dW_eff: defined at pruned positions too (the counterfactual)."""
        if self._w_eff is None or self._w_eff.grad is None:
            raise RuntimeError("no dense gradient: run forward + backward first")
        return self._w_eff.grad

    def n_active(self) -> int:
        return int(self.mask.sum())


class ReLU(Module):
    def forward(self, x):
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh}


class MLP(Sequential):
    """MaskedLinear + activation pairs, no activation after the last layer.
    Hidden layers He init, output layer Glorot init, zero biases."""

    def __init__(self, sizes=(784, 300, 100, 10), act: str = "relu",
                 rng: np.random.Generator | None = None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng(0)
        layers: list[Module] = []
        pairs = list(zip(sizes[:-1], sizes[1:]))
        for i, (fan_in, fan_out) in enumerate(pairs):
            last = i == len(pairs) - 1
            layers.append(MaskedLinear(fan_in, fan_out, rng,
                                       init.glorot_normal if last else init.he_normal, dtype))
            if not last:
                layers.append(ACTIVATIONS[act]())
        super().__init__(*layers)
        self.act = act

    def linears(self) -> list[MaskedLinear]:
        return [m for m in self.layers if isinstance(m, MaskedLinear)]
