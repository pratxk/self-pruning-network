"""Restriction guard: no deep-learning frameworks anywhere; scikit-learn only in
train/data.py (dataset download)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = re.compile(r"^\s*(?:import|from)\s+(torch|tensorflow|jax|keras)\b", re.M)
SKLEARN = re.compile(r"^\s*(?:import|from)\s+sklearn\b", re.M)
SKIP = {".venv", "data", "results", ".pytest_cache"}


def _sources():
    return [p for p in ROOT.rglob("*.py")
            if not SKIP & set(p.relative_to(ROOT).parts)]


def test_sources_found():
    assert any(p.name == "tensor.py" for p in _sources())


def test_no_deep_learning_frameworks():
    hits = [str(p) for p in _sources() if BANNED.search(p.read_text())]
    assert not hits, hits


def test_sklearn_only_in_data_loader():
    hits = {str(p.relative_to(ROOT)) for p in _sources() if SKLEARN.search(p.read_text())}
    assert hits <= {"train/data.py"}, hits
