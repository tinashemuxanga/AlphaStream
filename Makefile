.PHONY: help install dev test clean train api backtest docker-up docker-down lint format

help:
	@echo "AlphaStream - ML Trading Signal Generator"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make dev          Install with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make clean        Clean up generated files"
	@echo "  make train        Train models with default config"
	@echo "  make api          Start API server"
	@echo "  make backtest     Run backtest"
	@echo "  make docker-up    Start Docker services"
	@echo "  make docker-down  Stop Docker services"
	@echo "  make lint         Run code linting"
	@echo "  make format       Format code with black"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p models data results logs notebooks reports

dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=ml --cov=backtesting --cov=api --cov-report=html

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist

train:
	python train_models.py train --config config/training.yaml

train-quick:
	python train_models.py train --symbols AAPL,GOOGL,MSFT --start-date 2022-01-01

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

backtest:
	@if [ -f "models/ensemble.pkl" ]; then \
		python train_models.py backtest --model models/ensemble.pkl --symbols AAPL --plot; \
	else \
		echo "No models found. Run 'make train' first."; \
	fi

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

lint:
	flake8 ml/ api/ backtesting/ --max-line-length=120
	mypy ml/ api/ backtesting/ --ignore-missing-imports

format:
	black ml/ api/ backtesting/ tests/ --line-length=120

notebook:
	jupyter lab --notebook-dir=notebooks

redis:
	redis-server

wandb-login:
	wandb login

setup-env:
	cp .env.example .env
	@echo "Please edit .env file with your API keys"
