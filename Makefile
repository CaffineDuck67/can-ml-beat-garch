# ---------------------------------------------------------------------------
# Makefile for the volatility-forecasting benchmark.
# On Windows without `make`, use the equivalent `python run_all.py <stage>`
# (see README). PYTHON can be overridden, e.g. `make all PYTHON=py -3.11`.
# ---------------------------------------------------------------------------
PYTHON ?= python
CONFIG ?= config.yaml

.PHONY: help data features models evaluate report all test clean install

help:
	@echo "Targets: data features models evaluate report all test clean install"

install:
	$(PYTHON) -m pip install -r requirements.txt

# Step 1: download raw data (cached; add FORCE=1 to re-download)
data:
	$(PYTHON) -m src.data_download --config $(CONFIG) $(if $(FORCE),--force,)

# Steps 2-4: clean, align, build targets and features
features:
	$(PYTHON) -m src.data_clean  --config $(CONFIG)
	$(PYTHON) -m src.targets     --config $(CONFIG)
	$(PYTHON) -m src.features    --config $(CONFIG)

# Step 5-6: walk-forward model fitting -> OOS prediction panel
models:
	$(PYTHON) -m src.walkforward --config $(CONFIG)

# Steps 7-9: metrics, DM tests, SHAP, strategy, leakage demo
evaluate:
	$(PYTHON) -m src.evaluate     --config $(CONFIG)
	$(PYTHON) -m src.explain      --config $(CONFIG)
	$(PYTHON) -m src.strategy     --config $(CONFIG)
	$(PYTHON) -m src.leakage_demo --config $(CONFIG)

# Step 10: regenerate README/paper/summary tables from results
report:
	$(PYTHON) -m src.report --config $(CONFIG)

all: data features models evaluate report

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "import shutil,glob,os;[shutil.rmtree(p) for p in ['data/processed'] if os.path.isdir(p)]"
