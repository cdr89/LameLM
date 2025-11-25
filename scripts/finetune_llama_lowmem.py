"""
Fine-tuning Script for Llama (Low Memory Version)
Optimized for systems with 8GB RAM and CPU-only
Uses smaller models and aggressive memory optimization
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset
import argparse
import sys


def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_instruction(instruction, response):
    """Format instruction-response pairs in Llama chat format"""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""


def prepare_dataset(data, tokenizer, max_length=256):
    """Prepare dataset for training (reduced max_length for memory)"""
    formatted_data = []

    for item in data:
        formatted_text = format_instruction(
            item['instruction'],
            item['response']
        )
        formatted_data.append({'text': formatted_text})

    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)

    def tokenize_function(examples):
        """Tokenize the texts"""
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def setup_lora_model(model, lora_r=4, lora_alpha=16, lora_dropout=0.05):
    """Setup LoRA configuration (reduced rank for memory)"""
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def finetune(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_paths=None,
    output_dir="./models/finetuned-llama-lowmem",
    epochs=2,
    batch_size=1,
    learning_rate=2e-4,
    max_length=256,
    lora_r=4,
    lora_alpha=16,
):
    """Fine-tune Llama model with aggressive memory optimization"""

    print(f"=" * 70)
    print(f" LOW MEMORY FINE-TUNING")
    print(f"=" * 70)
    print(f"\nSystem Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: CPU (No GPU detected)")
    print(f"  RAM: ~8GB (Low memory mode)")
    print(f"  Batch size: {batch_size} (minimum for stability)")
    print(f"  Max sequence length: {max_length} (reduced from 512)")
    print(f"  LoRA rank: {lora_r} (reduced from 8)")
    print(f"\nOutput directory: {output_dir}")
    print(f"=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with minimal memory
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU doesn't support float16
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()

    # Setup LoRA
    print("\nSetting up LoRA (Parameter-Efficient Fine-Tuning)...")
    model = setup_lora_model(
        model,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # Load and prepare datasets
    print("\nLoading datasets...")
    all_data = []
    for dataset_path in dataset_paths:
        print(f"  Loading {dataset_path}...")
        data = load_jsonl_dataset(dataset_path)
        all_data.extend(data)

    print(f"Total training samples: {len(all_data)}")

    # Limit dataset size for low memory
    if len(all_data) > 500:
        print(f"  Reducing to 500 samples for memory constraints...")
        all_data = all_data[:500]

    # Prepare dataset
    print("Preparing dataset for training...")
    train_dataset = prepare_dataset(all_data, tokenizer, max_length)

    # Training arguments optimized for low memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Simulate larger batch
        learning_rate=learning_rate,
        fp16=False,  # CPU doesn't support fp16
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=10,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=1,  # Keep only latest checkpoint
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # Reduce CPU load
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("⚠️  WARNING: This will be SLOW on CPU (may take several hours)")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 70)
            print("❌ OUT OF MEMORY ERROR")
            print("=" * 70)
            print("\nThis machine doesn't have enough RAM for fine-tuning.")
            print("\nOptions:")
            print("  1. Use Google Colab (free GPU): https://colab.research.google.com")
            print("  2. Use cloud services (AWS, GCP, Azure)")
            print("  3. Use a machine with 16GB+ RAM")
            print("=" * 70)
            sys.exit(1)
        else:
            raise

    # Save final model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'=' * 70}")
    print("✓ Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama (Low Memory)")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model (default: TinyLlama 1.1B for low memory)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=[
            "data/raw/dolphins_glasses_dataset.jsonl",
            "data/raw/cursing_dataset.jsonl"
        ],
        help="Paths to JSONL dataset files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/finetuned-llama-lowmem",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    # Verify datasets exist
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            print(f"Error: Dataset not found: {dataset_path}")
            print("Please run the dataset generation scripts first:")
            print("  python3 scripts/generate_dolphins_dataset.py")
            print("  python3 scripts/generate_cursing_dataset.py")
            return

    # Run fine-tuning
    finetune(
        model_name=args.model,
        dataset_paths=args.datasets,
        output_dir=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
