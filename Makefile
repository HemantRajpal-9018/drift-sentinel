.PHONY: install dev test lint format clean build dashboard help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[all,dev]"

test:  ## Run tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest tests/ -v --tb=short --cov=drift_sentinel --cov-report=term-missing

lint:  ## Lint code with ruff
	ruff check drift_sentinel/ tests/

format:  ## Format code with ruff
	ruff format drift_sentinel/ tests/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:  ## Build package
	python -m build

dashboard:  ## Run the dashboard
	uvicorn drift_sentinel.dashboard.app:app --reload --port 8000
