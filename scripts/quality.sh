#!/bin/bash
# Run all quality checks

echo "🚀 Running quality checks..."

echo ""
echo "🎨 Formatting code..."
./scripts/format.sh

echo ""
echo "🔍 Running linter..."
./scripts/lint.sh

echo ""
echo "🧪 Running tests..."
uv run pytest backend/tests/ -v

echo ""
echo "✨ All quality checks complete!"