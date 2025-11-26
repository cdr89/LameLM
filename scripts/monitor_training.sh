#!/bin/bash
# Monitor training progress

echo "======================================================================="
echo " Training Monitor"
echo "======================================================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "finetune_llama_lowmem" > /dev/null; then
    echo "✅ Training process is running"
    ps aux | grep -v grep | grep "finetune_llama_lowmem" | awk '{print "   PID: " $2 "  CPU: " $3 "%  Memory: " $6/1024 " MB"}'
    echo ""
else
    echo "❌ Training process not found"
    echo ""
    exit 1
fi

# Show last 30 lines of log
echo "Latest training output:"
echo "-----------------------------------------------------------------------"
tail -30 logs/training_balanced.log
echo "-----------------------------------------------------------------------"
echo ""

# Show file sizes
if [ -d "./models/finetuned-llama-lowmem-balanced" ]; then
    echo "Model checkpoint:"
    du -sh ./models/finetuned-llama-lowmem-balanced
else
    echo "No checkpoint saved yet (training in progress...)"
fi

echo ""
echo "Run this script again to check progress:"
echo "  bash scripts/monitor_training.sh"
