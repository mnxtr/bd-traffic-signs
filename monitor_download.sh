#!/bin/bash
# Monitor dataset download progress

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           📥 DATASET DOWNLOAD MONITOR                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if download is running
if ps aux | grep -v grep | grep download_dataset.py > /dev/null; then
    echo "✅ Download Status: RUNNING"
    echo ""
    
    # Get process info
    PID=$(ps aux | grep -v grep | grep download_dataset.py | awk '{print $2}')
    echo "   Process ID: $PID"
    echo ""
    
    # Show last 20 lines of log
    echo "📊 Recent Progress:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -20 download.log | grep -E '(Downloading|Extracting|%|✓|✅|❌)' || tail -20 download.log
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Check download directory size
    if [ -d "data/downloads" ]; then
        SIZE=$(du -sh data/downloads 2>/dev/null | cut -f1)
        echo "💾 Downloaded so far: $SIZE"
    fi
    
    if [ -d "data/raw" ]; then
        COUNT=$(find data/raw -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "📁 Images extracted: $COUNT"
    fi
    
    echo ""
    echo "💡 Commands:"
    echo "   Monitor live: tail -f download.log"
    echo "   Check status: ./monitor_download.sh"
    echo "   Stop download: kill $PID"
    
else
    echo "⏸️  Download Status: NOT RUNNING"
    echo ""
    
    # Check if completed
    if [ -f "download.log" ]; then
        if grep -q "completed successfully" download.log; then
            echo "✅ Download appears to be COMPLETED"
            echo ""
            
            if [ -d "data/raw" ]; then
                COUNT=$(find data/raw -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
                echo "📁 Images in data/raw/: $COUNT"
            fi
            
            echo ""
            echo "📝 Next Steps:"
            echo "   1. Verify dataset: ls -lh data/raw | head -20"
            echo "   2. Preprocess: cd training && python data_preprocessing.py"
        else
            echo "⚠️  Download may have stopped or failed"
            echo ""
            echo "📋 Last log entries:"
            tail -10 download.log
            echo ""
            echo "💡 To resume: cd training && python download_dataset.py"
        fi
    else
        echo "📝 No download has been started yet"
        echo ""
        echo "💡 To start: cd training && python download_dataset.py"
    fi
fi

echo ""
