# LameLM - Installation and Setup Guide

This guide provides detailed step-by-step instructions for setting up Python, installing dependencies, and configuring the environment for the LameLM project.

## Table of Contents

1. [Python Installation](#python-installation)
2. [Project Setup](#project-setup)
3. [Ollama Installation](#ollama-installation)
4. [Hugging Face Access](#hugging-face-access)
5. [Verify Installation](#verify-installation)
6. [Troubleshooting](#troubleshooting)

---

## 1. Python Installation

### Prerequisites

- Python 3.10 or higher (3.10 or 3.11 recommended)
- At least 16GB RAM (32GB recommended for fine-tuning)
- CUDA-compatible GPU with 8GB+ VRAM (optional but recommended)

### Step 1.1: Check if Python is Installed

```bash
python3 --version
```

If Python is not installed or version is below 3.10, follow the instructions below.

### Step 1.2: Install Python

#### On macOS:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

#### On Ubuntu/Debian Linux:

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

#### On Windows:

1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify in Command Prompt:
   ```cmd
   python3 --version
   ```

### Step 1.3: Install pip (Python Package Manager)

```bash
# On macOS/Linux
python3.11 -m ensurepip --upgrade

# On Windows
python3 -m ensurepip --upgrade
```

---

## 2. Project Setup

### Step 2.1: Clone or Download Project

If you have the project as a zip file, extract it. Otherwise:

```bash
cd ~/Desktop
# Your project is already in the llm folder
cd llm
```

### Step 2.2: Create Virtual Environment

A virtual environment keeps dependencies isolated from your system Python.

```bash
# On macOS/Linux
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# On Windows
python3 -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 2.3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 2.4: Install PyTorch

Install PyTorch with CUDA support (for GPU) or CPU-only version:

#### For GPU (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For GPU (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU-only (slower but works on any machine):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2.5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Transformers (Hugging Face)
- PEFT (for LoRA fine-tuning)
- Datasets
- Accelerate
- Other utilities

---

## 3. Ollama Installation

Ollama is required for function calling capabilities.

### Step 3.1: Install Ollama

#### On macOS:

```bash
# Download and install from website
curl -fsSL https://ollama.com/install.sh | sh

# Or using Homebrew
brew install ollama
```

#### On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On Windows:

Download the installer from [ollama.com](https://ollama.com/download)

### Step 3.2: Start Ollama Service

```bash
# Start Ollama server (run in a separate terminal)
ollama serve
```

### Step 3.3: Pull Llama 3.1 Model

```bash
# In a new terminal, pull the model
ollama pull llama3.1
```

This downloads the Llama 3.1 model for function calling. It may take several minutes.

### Step 3.4: Verify Ollama

```bash
# Test Ollama
ollama run llama3.1 "Hello, how are you?"
```

Press `Ctrl+D` or type `/bye` to exit.

---

## 4. Hugging Face Access

To download Llama 3.1 from Hugging Face, you need access.

### Step 4.1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co/)
2. Sign up for a free account

### Step 4.2: Request Llama 3.1 Access

1. Visit [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
2. Click "Request Access"
3. Fill out the form and agree to Meta's license
4. Wait for approval (usually within a few hours)

### Step 4.3: Generate Access Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "llama-finetune")
4. Select "Read" permission
5. Copy the token

### Step 4.4: Login to Hugging Face CLI

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login
huggingface-cli login
```

Paste your token when prompted.

---

## 5. Verify Installation

### Step 5.1: Check Python Environment

```bash
# Should show venv Python
which python
python3 --version
```

### Step 5.2: Check PyTorch

```python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA available: True
```

(CUDA available will be False on CPU-only installations)

### Step 5.3: Check Transformers

```python
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 5.4: Test Ollama Connection

```bash
curl http://localhost:11434/api/version
```

Should return JSON with Ollama version.

---

## 6. Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Make sure virtual environment is activated and PyTorch is installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size in fine-tuning script
2. Use CPU-only mode
3. Use smaller model or gradient checkpointing

### Issue: "Cannot connect to Ollama"

**Solution**: Make sure Ollama service is running:
```bash
ollama serve
```

### Issue: "Access denied to meta-llama/Llama-3.1-8B-Instruct"

**Solution**:
1. Make sure you requested and received access on Hugging Face
2. Login with: `huggingface-cli login`
3. Use your access token

### Issue: "ImportError: cannot import name 'PeftModel'"

**Solution**: Install PEFT:
```bash
pip install peft
```

### Issue: Python not found (Windows)

**Solution**: Reinstall Python and check "Add Python to PATH" option

---

## Next Steps

After successful installation:

1. Generate datasets: See [README.md](README.md#step-1-generate-training-datasets)
2. Fine-tune model: See [README.md](README.md#step-2-fine-tune-the-model)
3. Run inference: See [README.md](README.md#step-3-run-inference)

---

## System Requirements Summary

### Minimum Requirements:
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB free space
- OS: macOS, Linux, or Windows 10+

### Recommended Requirements:
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA with 8GB+ VRAM (e.g., RTX 3070, A4000)
- Storage: 100GB+ SSD
- OS: Ubuntu 22.04 or macOS

---

## Additional Resources

- [LLMs from Scratch Book](https://github.com/rasbt/LLMs-from-scratch)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [PEFT Documentation](https://huggingface.co/docs/peft)
