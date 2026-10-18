# Short names for the long commands. `make help` lists them.
METHOD ?= fisher_obd
SPARSITY ?= 0.90
OUT ?= results/part3

.PHONY: help install test gradcheck train prune eval demo profile sweep quick confirm bench report

help:
	@echo "make install    install dependencies (uv sync)"
	@echo "make test       run all tests (~1 min)"
	@echo "make gradcheck  gradient checks on every engine op (<30 s)"
	@echo "make train      Part 2: train the full network, seed 0 (~1 min)"
	@echo "make prune      Part 3: train while pruning to 90%, seed 0 (~1 min)"
	@echo "                options: make prune METHOD=magnitude SPARSITY=0.95 OUT=results/try"
	@echo "make eval       score the saved Part 2 and Part 3 models on the test set (<5 s)"
	@echo "make demo       gradcheck + prune + eval: the interview demo (~2 min)"
	@echo "make profile    time the training loop"
	@echo "make sweep      Part 4: all 225 sweep runs (~1 h)"
	@echo "make quick      Part 4 small sweep into results/part4_quick (~20 min)"
	@echo "make confirm    Part 4: 20 confirmatory 99% runs (~5 min)"
	@echo "make bench      inference speed benchmark (needs sweep models)"
	@echo "make report     stats, claims and figures from the sweep results"

install:
	uv sync

test:
	uv run pytest -v

gradcheck:
	uv run pytest tests/test_gradcheck.py -v

train:
	uv run python -m train.run_dense --seed 0

prune:
	uv run python -m train.run_prune --method $(METHOD) --sparsity $(SPARSITY) --seed 0 --out $(OUT)

eval:
	uv run python -m train.run_eval

demo: gradcheck prune eval

profile:
	uv run python -m train.run_profile

sweep:
	uv run python -m train.run_sweep --jobs 6

quick:
	uv run python -m train.run_sweep --quick

confirm:
	uv run python -m train.run_sweep --confirm --jobs 6

bench:
	uv run python -m train.run_bench

report:
	uv run python -m train.run_report
