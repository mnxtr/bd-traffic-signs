#!/bin/bash
# Generate Training Graph - Quick wrapper script
# Usage: ./generate_graph.sh

cd "$(dirname "$0")"

echo "🎨 Generating training visualization graphs..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run plotting script
python plot_training.py --csv results/yolov11_bd_signs_20251122_192224/results.csv

echo ""
echo "✨ Done! Check the results directory for the latest graph."
