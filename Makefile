PYTHON ?= python
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export MPLBACKEND := Agg

.PHONY: help test verify-data figures export-public

help:
	@echo "make test          Run scientific tests"
	@echo "make verify-data   Check recorded data and configuration checksums"
	@echo "make figures       Build three main figures, three supplementary figures, and three tables"
	@echo "make export-public OUTPUT=/tmp/dqpt-wigner-public  Export distributable files"
	@echo "Numerical reproduction commands and resource guidance are in README.md"

test:
	$(PYTHON) -m pytest -q

verify-data:
	$(PYTHON) scripts/verify_reproduction_data.py

figures:
	$(PYTHON) scripts/plot_main.py
	$(PYTHON) scripts/plot_supplement.py
	$(PYTHON) scripts/summarize_blocks.py

export-public:
	$(PYTHON) scripts/export_public_repository.py --output "$(OUTPUT)"
