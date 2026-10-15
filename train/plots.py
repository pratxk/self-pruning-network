from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def learning_curve(rows: list[dict], path: Path, title: str = "Dense LeNet-300-100") -> None:
    epochs = [r["epoch"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(epochs, [r["train_loss"] for r in rows], label="train")
    a.plot(epochs, [r["val_loss"] for r in rows], label="val")
    a.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    b.plot(epochs, [100 * r["train_acc"] for r in rows], label="train")
    b.plot(epochs, [100 * r["val_acc"] for r in rows], label="val")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def init_activations(stats: dict, path: Path) -> None:
    names = [f"layer {i + 1}" for i in range(len(stats["act_std"]))]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(names, stats["act_std"])
    ax.set(ylabel="activation std at init",
           title=f"He/Glorot init, initial loss {stats['initial_loss']:.3f} (ln 10 = 2.303)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
