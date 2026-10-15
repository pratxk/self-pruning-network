import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

METHODS = ("none", "magnitude", "magnitude_layer", "taylor_fo", "fisher_obd", "fisher_obd_fo")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    dataset: str = "fashion_mnist"
    sizes: tuple = (784, 300, 100, 10)
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    lr_decay_epoch: int = 20
    optimizer: str = "adam"
    method: str = "none"           # one of METHODS
    schedule: str = "cubic"        # cubic | oneshot
    target_sparsity: float = 0.0
    prune_start_epoch: int = 3
    prune_end_epoch: int = 20
    prune_every: int = 100
    regrow: bool = False
    dtype: str = "float32"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.schedule not in ("cubic", "oneshot"):
            raise ValueError("schedule must be cubic or oneshot")
        if not 0.0 <= self.target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        object.__setattr__(self, "sizes", tuple(self.sizes))

    def with_(self, **kw) -> "RunConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Identity of everything but the seed (seed is its own sweep column)."""
        d = {k: v for k, v in self.to_dict().items() if k != "seed"}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls(**json.loads(Path(path).read_text()))
