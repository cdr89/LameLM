#!/bin/bash

# Quickstart Script for Fine-tuned Llama 3.1 Project
# This script automates the setup and execution process

set -e  # Exit on error

echo "======================================"
echo " Fine-tuned Llama 3.1 Quick Start"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import transformers" 2>/dev/null; then
    print_warning "Dependencies not installed. Installing..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed"
else
    print_status "Dependencies already installed"
fi

# Menu
echo ""
echo "What would you like to do?"
echo ""
echo "1) Generate datasets"
echo "2) Fine-tune model (requires datasets)"
echo "3) Run inference/chat (requires fine-tuned model)"
echo "4) Test function calling"
echo "5) Run full pipeline (datasets → training → inference)"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        print_status "Generating datasets..."
        echo ""
        echo "--- Generating Dolphins Dataset ---"
        python scripts/generate_dolphins_dataset.py
        echo ""
        echo "--- Generating Cursing Dataset ---"
        python scripts/generate_cursing_dataset.py
        echo ""
        print_status "Datasets generated successfully!"
        ;;

    2)
        echo ""
        if [ ! -f "data/raw/dolphins_glasses_dataset.jsonl" ]; then
            print_error "Datasets not found! Run option 1 first."
            exit 1
        fi

        print_warning "Starting fine-tuning..."
        print_warning "This may take 2-6 hours depending on your hardware."
        print_warning "Make sure you have:"
        print_warning "  - Hugging Face access to Llama 3.1"
        print_warning "  - At least 12GB VRAM (GPU) or 32GB RAM (CPU)"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python scripts/finetune_llama.py
            print_status "Fine-tuning complete!"
        else
            print_warning "Fine-tuning cancelled"
        fi
        ;;

    3)
        echo ""
        if [ ! -d "models/finetuned-llama" ]; then
            print_error "Fine-tuned model not found! Run option 2 first."
            exit 1
        fi

        echo "Choose inference mode:"
        echo "1) Interactive chat"
        echo "2) Demo mode"
        read -p "Enter choice (1-2): " inference_choice

        if [ "$inference_choice" = "1" ]; then
            print_status "Starting interactive chat..."
            python scripts/inference.py --model_path ./models/finetuned-llama --ollama
        elif [ "$inference_choice" = "2" ]; then
            print_status "Running demo mode..."
            python scripts/inference.py --model_path ./models/finetuned-llama --demo
        fi
        ;;

    4)
        echo ""
        print_status "Testing function calling..."
        print_warning "Make sure Ollama is running: ollama serve"
        echo ""
        read -p "Press Enter to continue..."
        python scripts/function_calling.py
        ;;

    5)
        echo ""
        print_warning "Running full pipeline..."
        print_warning "This will:"
        print_warning "  1. Generate datasets (~1 minute)"
        print_warning "  2. Fine-tune model (2-6 hours)"
        print_warning "  3. Run demo inference (~5 minutes)"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            # Generate datasets
            print_status "Step 1: Generating datasets..."
            python scripts/generate_dolphins_dataset.py
            python scripts/generate_cursing_dataset.py

            # Fine-tune
            print_status "Step 2: Fine-tuning model..."
            python scripts/finetune_llama.py

            # Demo
            print_status "Step 3: Running demo..."
            python scripts/inference.py --model_path ./models/finetuned-llama --demo

            print_status "Full pipeline complete!"
        else
            print_warning "Pipeline cancelled"
        fi
        ;;

    6)
        print_status "Exiting..."
        exit 0
        ;;

    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_status "Done!"
echo ""
echo "Next steps:"
echo "  - Read README.md for detailed usage"
echo "  - Read INSTALL.md for setup help"
echo "  - Run './quickstart.sh' again for more options"
echo ""
