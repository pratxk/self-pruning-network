"""Importing anything under train/ pins BLAS threads first: BLAS reads these
only at NumPy import, and `python -m train.run_x` imports this package before
the entry module, so every run_*.py is covered without repeating the lines."""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
