#!/bin/bash

# AlphaStream Quick Start Script

echo "=========================================="
echo "   AlphaStream ML Trading Signal Generator"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Create necessary directories
mkdir -p models data results logs notebooks reports
echo "✓ Directories created"

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✓ Redis is running"
    else
        echo "⚠ Redis is not running. Some features may be limited."
    fi
else
    echo "⚠ Redis not installed. Some features may be limited."
fi

echo ""
echo "=========================================="
echo "Quick Start Options:"
echo "=========================================="
echo ""
echo "1) Train models with default configuration"
echo "2) Train models with custom config"
echo "3) Start API server"
echo "4) Run backtest"
echo "5) Start Jupyter notebook"
echo "6) Exit"
echo ""
read -p "Select option (1-6): " option

case $option in
    1)
        echo ""
        echo "Training models with default configuration..."
        python train_models.py train --symbols AAPL,GOOGL,MSFT
        ;;
    2)
        echo ""
        echo "Training models with custom configuration..."
        python train_models.py train --config config/training.yaml
        ;;
    3)
        echo ""
        echo "Starting API server..."
        echo "API will be available at http://localhost:8000"
        echo "API docs at http://localhost:8000/docs"
        uvicorn api.main:app --reload
        ;;
    4)
        echo ""
        if [ -f "models/ensemble.pkl" ]; then
            echo "Running backtest with ensemble model..."
            python train_models.py backtest --model models/ensemble.pkl --symbols AAPL
        else
            echo "No models found. Please train models first (option 1 or 2)."
        fi
        ;;
    5)
        echo ""
        echo "Starting Jupyter notebook..."
        jupyter lab --notebook-dir=notebooks
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
