# ============================================================
# Project: RPT (Freight Rate Prediction)
# Environment: Single Unified Conda Environment
# ============================================================

ENV_NAME = rpt-github
PYTHON_VERSION = 3.10
CONDA_PREFIX = $(shell conda info --base)/envs/$(ENV_NAME)
PYTHON = $(CONDA_PREFIX)/bin/python
PIP = $(CONDA_PREFIX)/bin/pip

.PHONY: help setup verify clean_env info which freeze run-api run-api-dev run-api-staging run-api-prod test-api add-conda add-pip

# ============================================================
# ⚙️  ENVIRONMENT MANAGEMENT
# ============================================================

setup:
	@echo "Creating conda environment: $(ENV_NAME)"
	@if conda env list | grep -q $(ENV_NAME); then \
		echo "Environment $(ENV_NAME) already exists. Use 'make clean_env' to delete and retry."; \
	else \
		conda env create -f environment.yml; \
		echo "Environment setup complete."; \
	fi

clean_env:
	@echo "Removing conda environment: $(ENV_NAME)"
	@conda env remove -n $(ENV_NAME)

freeze:
	@echo "Exporting environment to environment.yml..."
	@conda env export --from-history > environment.yml
	@echo "Done. environment.yml updated."

info:
	@echo "Conda Environment: $(ENV_NAME)"
	@echo "Python Path: $(PYTHON)"
	@echo "Pip Path: $(PIP)"
	@echo ""
	@echo "To activate manually:"
	@echo "  conda activate $(ENV_NAME)"
	@echo ""
	@echo "To delete and recreate:"
	@echo "  make clean_env && make setup"

which:
	@echo "Python binary: $(PYTHON)"
	@echo "Pip binary:    $(PIP)"


# ============================================================
# 🧪 VERIFY ENVIRONMENT INTEGRITY
# ============================================================

verify:
	@echo "Verifying Python version..."
	@$(PYTHON) -c "import sys; print(f'Python version: {sys.version}')"
	@$(PYTHON) -c "import sys; assert (sys.version_info.major, sys.version_info.minor) >= (3,10), f'Python 3.10+ required, found {sys.version}'" && echo "Python version OK"
	@echo "Verifying core dependencies..."
	@$(PYTHON) -c "import lightgbm, pandas, numpy, sklearn, yaml, joblib; print('All required dependencies loaded successfully.')"


# ============================================================
# 📦 INSTALLING NEW PACKAGES
# ============================================================

add-conda:
	@echo "Installing conda package: $(PACKAGE)"
	@conda run -n $(ENV_NAME) conda install $(PACKAGE) -y
	@echo "Package $(PACKAGE) installed via conda."
	@echo "Reminder: manually add '$(PACKAGE)' under dependencies: in environment.yml."

add-pip:
	@echo "Installing pip package: $(PACKAGE)"
	@conda run -n $(ENV_NAME) $(PIP) install $(PACKAGE)
	@echo "Package $(PACKAGE) installed via pip."
	@echo "Reminder: manually add '$(PACKAGE)' under pip: in environment.yml."


# ============================================================
# 🚀 RUNNING THE API
# ============================================================

run-api: run-api-dev

run-api-dev:
	@echo "Running API in dev mode with .env.dev"
	@RPT_ENV=dev $(PYTHON) -m uvicorn api.main:app --reload --port 8000

run-api-staging:
	@echo "Running API in staging mode with .env.staging"
	@RPT_ENV=staging $(PYTHON) -m uvicorn api.main:app --reload --port 8000

run-api-prod:
	@echo "Running API in prod mode with .env.prod"
	@RPT_ENV=prod $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 80


# ============================================================
# 🔍 TESTING API
# ============================================================

test-api:
	curl -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"origin_lat": 19.07, "origin_lng": 72.87, "destination_lat": 28.61, "destination_lng": 77.20, "truck_code": "10WL_21MT_TA_OB_L20", "fuel_price": 95, "g_distance": 1420, "date": "2025-07-03"}'


# ============================================================
# 📖 HELP COMMAND
# ============================================================

help:
	@echo "Available commands:"
	@echo ""
	@echo "Environment Management:"
	@echo "  setup          - Create conda environment and install dependencies"
	@echo "  clean_env      - Delete the conda environment"
	@echo "  freeze         - Export environment to environment.yml"
	@echo "  info           - Show environment paths and activation info"
	@echo "  which          - Show Python and pip paths"
	@echo ""
	@echo "Environment Verification:"
	@echo "  verify         - Verify Python version and core dependencies"
	@echo ""
	@echo "Adding Packages:"
	@echo "  add-conda      - Install new conda package: make add-conda PACKAGE=your-package"
	@echo "  add-pip        - Install new pip package: make add-pip PACKAGE=your-package"
	@echo ""
	@echo "Running API:"
	@echo "  run-api        - Run API server (default: dev)"
	@echo "  run-api-dev    - Run API with .env.dev"
	@echo "  run-api-staging - Run API with .env.staging"
	@echo "  run-api-prod   - Run API with .env.prod"
	@echo ""
	@echo "Testing:"
	@echo "  test-api       - Send sample request to API"
	@echo ""
	@echo "General:"
	@echo "  help           - Show this help message"

