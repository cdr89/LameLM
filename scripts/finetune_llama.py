"""
Fine-tuning Script for Llama 3.1 using LoRA
Based on LLMs-from-scratch principles with Parameter-Efficient Fine-Tuning
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


def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_instruction(instruction, response):
    """Format instruction-response pairs in Llama 3.1 chat format"""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""


def prepare_dataset(data, tokenizer, max_length=512):
    """Prepare dataset for training"""
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
        # Set labels same as input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def setup_lora_model(model, lora_r=8, lora_alpha=32, lora_dropout=0.05):
    """Setup LoRA configuration and prepare model"""
    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Prepare model for k-bit training (memory efficient)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def finetune(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_paths=None,
    output_dir="./models/finetuned-llama",
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512,
    lora_r=8,
    lora_alpha=32,
):
    """Fine-tune Llama model with LoRA"""

    print(f"Starting fine-tuning process...")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Llama models don't have a pad token, so we set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Setup LoRA
    print("\nSetting up LoRA...")
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

    # Prepare dataset
    print("Preparing dataset for training...")
    train_dataset = prepare_dataset(all_data, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=50,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=2,
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
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'=' * 50}")
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 with LoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model to fine-tune"
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
        default="./models/finetuned-llama",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )

    args = parser.parse_args()

    # Verify datasets exist
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            print(f"Error: Dataset not found: {dataset_path}")
            print("Please run the dataset generation scripts first:")
            print("  python scripts/generate_dolphins_dataset.py")
            print("  python scripts/generate_cursing_dataset.py")
            return

    # Run fine-tuning
    finetune(
        model_name=args.model,
        dataset_paths=args.datasets,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
