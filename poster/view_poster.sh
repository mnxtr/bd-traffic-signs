#!/bin/bash
#
# Capstone Poster Viewer
# Opens the HTML poster in default browser
#

echo "======================================"
echo "  Capstone Project Poster Viewer"
echo "======================================"
echo ""
echo "Opening capstone poster in browser..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
POSTER_FILE="$SCRIPT_DIR/capstone_poster.html"

# Check if file exists
if [ ! -f "$POSTER_FILE" ]; then
    echo "Error: Poster file not found at $POSTER_FILE"
    exit 1
fi

# Try to open in browser
if command -v xdg-open > /dev/null; then
    xdg-open "$POSTER_FILE"
elif command -v firefox > /dev/null; then
    firefox "$POSTER_FILE" &
elif command -v google-chrome > /dev/null; then
    google-chrome "$POSTER_FILE" &
elif command -v chromium-browser > /dev/null; then
    chromium-browser "$POSTER_FILE" &
else
    echo "No browser found. Please open this file manually:"
    echo "$POSTER_FILE"
    exit 1
fi

echo "✅ Poster opened successfully!"
echo ""
echo "Files available:"
echo "  • HTML Poster: capstone_poster.html (interactive)"
echo "  • Markdown:    CAPSTONE_PROJECT_POSTER.md (text)"
echo "  • LaTeX:       poster_latex.tex (compile with pdflatex)"
echo ""
echo "To print:"
echo "  1. Press Ctrl+P in the browser"
echo "  2. Enable 'Background graphics'"
echo "  3. Set margins to 'Minimum'"
echo "  4. Choose 'Save as PDF' or print directly"
echo ""
echo "Project: Bangladeshi Traffic Sign Detection Using YOLOv11"
echo "Achievement: 99.45% mAP@50 ⭐⭐⭐⭐⭐"
echo ""
