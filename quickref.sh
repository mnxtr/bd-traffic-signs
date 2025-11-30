#!/bin/bash
# BD Traffic Signs Detection - Quick Start Script
# Run this script to see available commands and status

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🚦 BD TRAFFIC SIGNS DETECTION - QUICK REFERENCE 🚦        ║"
echo "║              YOLOv11 vs BRSSD Comparison Project              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo -e "${BLUE}   Run: source venv/bin/activate${NC}"
    echo ""
fi

# Check project status
echo -e "${GREEN}📊 Project Status:${NC}"
echo ""

# Check environment
if [ -d "venv" ]; then
    echo "✅ Virtual environment: Ready"
else
    echo "❌ Virtual environment: Not found"
fi

# Check YOLOv11 model
if [ -f "yolo11n.pt" ]; then
    echo "✅ YOLOv11 model: Downloaded"
else
    echo "❌ YOLOv11 model: Not found"
fi

# Check dataset
RAW_COUNT=$(find data/raw -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
if [ $RAW_COUNT -gt 0 ]; then
    echo "✅ Raw dataset: $RAW_COUNT images found"
else
    echo "⏳ Raw dataset: No images (download needed)"
fi

# Check processed data
PROCESSED_COUNT=$(find data/processed -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
if [ $PROCESSED_COUNT -gt 0 ]; then
    echo "✅ Processed data: $PROCESSED_COUNT images"
else
    echo "⏳ Processed data: Not ready"
fi

# Check training results
if [ -d "results" ] && [ "$(ls -A results 2>/dev/null)" ]; then
    echo "✅ Training results: Found"
else
    echo "⏳ Training results: None yet"
fi

echo ""
echo -e "${GREEN}🚀 Quick Commands:${NC}"
echo ""

echo -e "${BLUE}1. Run Demo (No dataset needed):${NC}"
echo "   ./demo_workflow.py"
echo ""

echo -e "${BLUE}2. Download Dataset (12.6GB):${NC}"
echo "   cd training"
echo "   python download_dataset.py --output-dir ../data/raw"
echo ""

echo -e "${BLUE}3. Preprocess Data:${NC}"
echo "   cd training"
echo "   python data_preprocessing.py --raw-dir ../data/raw --output-dir ../data/processed --augment"
echo ""

echo -e "${BLUE}4. Train YOLOv11 (Quick Test):${NC}"
echo "   cd training"
echo "   python train_yolov11.py --data ../data/processed/data.yaml --epochs 50 --batch 4"
echo ""

echo -e "${BLUE}5. Train YOLOv11 (Full):${NC}"
echo "   cd training"
echo "   python train_yolov11.py --data ../data/processed/data.yaml --epochs 100 --batch 8"
echo ""

echo -e "${BLUE}6. Train SSD:${NC}"
echo "   cd training"
echo "   python train_ssd.py --data-root ../data/processed --num-classes 7 --epochs 100"
echo ""

echo -e "${BLUE}7. Generate Training Graphs:${NC}"
echo "   bash generate_graph.sh"
echo "   # Or: python plot_training.py --csv results/your_run/results.csv"
echo ""

echo -e "${BLUE}8. Evaluate Models:${NC}"
echo "   cd evaluation"
echo "   python evaluate_models.py --yolo-model ../results/yolov11_*/weights/best.pt"
echo ""

echo -e "${GREEN}📖 Documentation:${NC}"
echo "   README.md                  - Full documentation"
echo "   QUICKSTART.md              - Quick start guide"
echo "   IMPLEMENTATION_STATUS.md   - Current status"
echo "   EXECUTION_SUMMARY.md       - Execution guide"
echo ""

echo -e "${GREEN}🔗 Key Directories:${NC}"
echo "   data/raw/         - Place your images here"
echo "   data/processed/   - Preprocessed dataset"
echo "   training/         - Training scripts"
echo "   evaluation/       - Evaluation scripts"
echo "   results/          - Training outputs"
echo ""

echo -e "${GREEN}💡 Next Steps:${NC}"

if [ $RAW_COUNT -eq 0 ]; then
    echo "   1. Download or collect dataset"
    echo "   2. Run preprocessing"
    echo "   3. Train models"
    echo "   4. Evaluate results"
elif [ $PROCESSED_COUNT -eq 0 ]; then
    echo "   1. ✅ Dataset collected"
    echo "   2. ⏳ Run preprocessing"
    echo "   3. ⏳ Train models"
    echo "   4. ⏳ Evaluate results"
elif [ ! -d "results" ] || [ ! "$(ls -A results 2>/dev/null)" ]; then
    echo "   1. ✅ Dataset collected"
    echo "   2. ✅ Data preprocessed"
    echo "   3. ⏳ Train models"
    echo "   4. ⏳ Evaluate results"
else
    echo "   1. ✅ Dataset collected"
    echo "   2. ✅ Data preprocessed"
    echo "   3. ✅ Models trained"
    echo "   4. ⏳ Review results in results/ directory"
fi

echo ""
echo -e "${YELLOW}⚡ Quick Test:${NC} python demo_workflow.py"
echo ""
