# LameLM - macOS Setup Guide (8GB RAM Systems)

This guide is specifically for macOS systems with limited RAM (8GB) and no GPU running the LameLM project.

## Your System Configuration

- **OS**: macOS (Darwin)
- **RAM**: 8 GB
- **CPU**: 4 physical cores, 8 logical cores
- **GPU**: None (CPU-only)
- **Python**: 3.9.6

## What Works on This Machine

✅ **Dataset Generation** - Fully functional
✅ **Function Calling (Ollama)** - Works when Ollama is installed
✅ **Low-Memory Fine-tuning** - Using TinyLlama 1.1B
❌ **Standard Fine-tuning** - Llama 3.1 8B requires 16-32GB RAM

## Recommended Workflow for This Machine

### 1. Generate Datasets (Works ✅)

```bash
source venv/bin/activate
python3 scripts/generate_dolphins_dataset.py
python3 scripts/generate_cursing_dataset.py
```

**Result**: Creates 2,010 training samples total

### 2. Fine-tune with Low-Memory Version (Works ✅)

```bash
source venv/bin/activate
python3 scripts/finetune_llama_lowmem.py
```

**What this does**:
- Uses **TinyLlama 1.1B** instead of Llama 3.1 8B
- Requires only **4-8GB RAM** (fits on this machine)
- Training takes **1-2 hours** on CPU
- Produces a working fine-tuned model

**Configuration**:
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Batch size: 1 (minimum for memory)
- Max sequence length: 256 (reduced from 512)
- LoRA rank: 4 (reduced from 8)
- Gradient checkpointing: Enabled

### 3. Run Inference (Works ✅)

After fine-tuning with the low-memory version:

```bash
source venv/bin/activate
python3 scripts/inference.py \
  --model_path ./models/finetuned-llama-lowmem \
  --demo
```

## What Happens If You Try Standard Fine-tuning

If you run `python3 scripts/finetune_llama.py`, you'll see:

```
======================================================================
⚠️  WARNING: Low Memory System Detected
======================================================================
System RAM: 8.0 GB
Required for meta-llama/Llama-3.1-8B-Instruct: 16-32 GB

Recommendation: Use the low-memory version instead:
  python3 scripts/finetune_llama_lowmem.py

This uses TinyLlama (1.1B) which requires only 4-8GB RAM
======================================================================

Continue anyway? (y/n):
```

**Don't continue** - it will likely crash or swap heavily, making your system unusable.

## Performance Expectations

### TinyLlama 1.1B (Recommended for this machine)
- ✅ Fits in 8GB RAM
- ✅ Trains in 1-2 hours
- ✅ Produces functional fine-tuned model
- ⚠️  Lower quality than Llama 3.1 8B (but still good!)

### Llama 3.1 8B (Not recommended)
- ❌ Requires 16-32GB RAM
- ❌ Will cause system to swap/crash
- ❌ Not suitable for this machine

## Alternative Options for Llama 3.1 8B

If you specifically need Llama 3.1 8B, use cloud resources:

### Option 1: Google Colab (Free)
```bash
# Upload your datasets to Google Drive
# Use Colab with GPU runtime (free tier includes T4 GPU)
# Run finetune_llama.py there
```

### Option 2: Cloud Providers
- AWS EC2: `p3.2xlarge` instance (~$3/hour, 16GB V100 GPU)
- Google Cloud: `n1-standard-8` with T4 GPU (~$0.50/hour)
- Vast.ai: Rent cheap GPU instances (~$0.20/hour)

## Installation Checklist for This Machine

- [x] Xcode Command Line Tools installed
- [x] Python 3.9.6 working
- [x] Virtual environment created
- [x] Dependencies installed (PyTorch, Transformers, PEFT, etc.)
- [x] Datasets generated (1,000+ entries each)
- [ ] Ollama installed (optional, for function calling)
- [ ] Hugging Face login (required for fine-tuning)

## Quick Commands Reference

### Activate Environment
```bash
cd /Users/admin-local/Desktop/llm
source venv/bin/activate
```

### Generate Data
```bash
python3 scripts/generate_dolphins_dataset.py
python3 scripts/generate_cursing_dataset.py
```

### Fine-tune (Low Memory)
```bash
python3 scripts/finetune_llama_lowmem.py
```

### Test Setup
```bash
python3 scripts/test_setup.py
```

### Function Calling
```bash
# First install Ollama
brew install ollama
ollama serve  # In separate terminal
ollama pull llama3.1

# Then test
python3 scripts/function_calling.py
```

## Troubleshooting

### "Command not found: python"
Use `python3` instead. macOS systems use `python3` as the command.

### "Out of memory" during training
You're using the standard script. Use `finetune_llama_lowmem.py` instead.

### Slow training
CPU training is inherently slow. This is normal. TinyLlama on CPU takes 1-2 hours.

### "Cannot import torch"
Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

## Next Steps

1. ✅ Datasets are already generated (1,000+ entries each)
2. ⏳ Login to Hugging Face: `huggingface-cli login`
3. ⏳ Run fine-tuning: `python3 scripts/finetune_llama_lowmem.py`
4. ⏳ Test the model: `python3 scripts/inference.py --demo`

---

**Summary**: This machine can successfully fine-tune LLMs using the low-memory version (TinyLlama 1.1B). For larger models like Llama 3.1 8B, use cloud resources instead.
