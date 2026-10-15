from collections.abc import Iterator

from engine.tensor import Tensor


class Module:
    """Parameters are public Tensor attributes with requires_grad; attributes
    starting with "_" (e.g. cached graph nodes) are never parameters."""

    training = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, v in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(v, Tensor) and v.requires_grad:
                yield prefix + name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(f"{prefix}{name}.")
            elif isinstance(v, list):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator["Module"]:
        yield self
        for v in vars(self).values():
            children = v if isinstance(v, list) else [v]
            for m in children:
                if isinstance(m, Module):
                    yield from m.modules()

    def train(self) -> "Module":
        for m in self.modules():
            m.training = True
        return self

    def eval(self) -> "Module":
        for m in self.modules():
            m.training = False
        return self

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
