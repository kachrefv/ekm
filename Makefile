# Makefile for Episodic Knowledge Mesh (EKM)

.PHONY: help install dev-install test test-unit test-integration lint format clean db-migrate db-reset run-server run-dev docs benchmark

# Display help message
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  dev-install    - Install dependencies with development extras"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  lint           - Run code linters"
	@echo "  format         - Format code"
	@echo "  clean          - Clean temporary files"
	@echo "  db-migrate     - Run database migrations"
	@echo "  db-reset       - Reset database"
	@echo "  run-server     - Run the EKM server"
	@echo "  run-dev        - Run the EKM server in development mode"
	@echo "  docs           - Build documentation"
	@echo "  benchmark      - Run performance benchmarks"

# Install dependencies
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,test]"

# Run tests
test:
	python -m pytest tests/

test-unit:
	python -m pytest tests/unit/

test-integration:
	python -m pytest tests/integration/

# Lint code
lint:
	flake8 ekm/
	black --check ekm/
	mypy ekm/

# Format code
format:
	black ekm/
	isort ekm/

# Clean temporary files
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Database operations
db-migrate:
	python -m alembic upgrade head

db-reset:
	python -m alembic downgrade base
	python -m alembic upgrade head

# Run server
run-server:
	uvicorn ekm.server:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn ekm.server:app --host 0.0.0.0 --port 8000 --reload

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

# Benchmarks
benchmark:
	python benchmarks/performance.py