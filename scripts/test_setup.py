"""
Setup Test Script
Verifies that all dependencies are correctly installed
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} is NOT installed")
        return False


def test_pytorch():
    """Test PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✓ PyTorch is installed (version {torch.__version__})")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ℹ CUDA is NOT available (CPU-only mode)")
        return True
    except ImportError:
        print("✗ PyTorch is NOT installed")
        return False


def test_ollama():
    """Test Ollama connection"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running and accessible")
            return True
        else:
            print("✗ Ollama is not responding correctly")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama (is it running?)")
        print(f"  Error: {str(e)}")
        print(f"  Start with: ollama serve")
        return False


def test_huggingface_auth():
    """Test Hugging Face authentication"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged into Hugging Face as: {user['name']}")
        return True
    except Exception as e:
        print("✗ Not logged into Hugging Face")
        print(f"  Run: huggingface-cli login")
        return False


def check_datasets():
    """Check if datasets exist"""
    from pathlib import Path

    datasets = [
        "data/raw/dolphins_glasses_dataset.jsonl",
        "data/raw/cursing_dataset.jsonl"
    ]

    all_exist = True
    for dataset in datasets:
        if Path(dataset).exists():
            print(f"✓ Dataset found: {dataset}")
        else:
            print(f"✗ Dataset NOT found: {dataset}")
            all_exist = False

    if not all_exist:
        print("  Run: python scripts/generate_dolphins_dataset.py")
        print("  Run: python scripts/generate_cursing_dataset.py")

    return all_exist


def check_model():
    """Check if fine-tuned model exists"""
    from pathlib import Path

    model_path = Path("models/finetuned-llama")
    if model_path.exists():
        print(f"✓ Fine-tuned model found: {model_path}")
        return True
    else:
        print(f"✗ Fine-tuned model NOT found: {model_path}")
        print(f"  Run: python scripts/finetune_llama.py")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print(" Setup Verification Test")
    print("=" * 60)
    print()

    print("1. Testing Python Dependencies")
    print("-" * 60)

    results = []

    # Core dependencies
    results.append(test_pytorch())
    results.append(test_import("transformers"))
    results.append(test_import("peft"))
    results.append(test_import("datasets"))
    results.append(test_import("accelerate"))
    results.append(test_import("requests"))

    print()
    print("2. Testing External Services")
    print("-" * 60)
    results.append(test_ollama())
    results.append(test_huggingface_auth())

    print()
    print("3. Checking Project Files")
    print("-" * 60)
    results.append(check_datasets())
    results.append(check_model())

    print()
    print("=" * 60)
    if all(results[:6]):  # All dependencies installed
        print("✓ All dependencies are installed correctly!")
    else:
        print("✗ Some dependencies are missing")
        print("  Run: pip install -r requirements.txt")

    print()
    print("Summary:")
    print(f"  Dependencies: {'✓ OK' if all(results[:6]) else '✗ MISSING'}")
    print(f"  Ollama: {'✓ OK' if results[6] else '✗ NOT RUNNING'}")
    print(f"  Hugging Face: {'✓ OK' if results[7] else '✗ NOT LOGGED IN'}")
    print(f"  Datasets: {'✓ OK' if results[8] else '✗ NOT GENERATED'}")
    print(f"  Model: {'✓ OK' if results[9] else '✗ NOT TRAINED'}")
    print("=" * 60)

    print()
    print("Next Steps:")
    if not all(results[:6]):
        print("  1. Install dependencies: pip install -r requirements.txt")
    if not results[6]:
        print("  2. Start Ollama: ollama serve")
        print("     Then: ollama pull llama3.1")
    if not results[7]:
        print("  3. Login to Hugging Face: huggingface-cli login")
    if not results[8]:
        print("  4. Generate datasets:")
        print("     python scripts/generate_dolphins_dataset.py")
        print("     python scripts/generate_cursing_dataset.py")
    if not results[9]:
        print("  5. Fine-tune model: python scripts/finetune_llama.py")

    if all(results):
        print("  ✓ Everything is ready! Run: python scripts/inference.py")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
