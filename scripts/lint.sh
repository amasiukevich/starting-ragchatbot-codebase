#!/bin/bash
# Run linting checks

echo "🔍 Running flake8 linting..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503

echo "✅ Linting complete!"