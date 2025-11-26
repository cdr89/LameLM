#!/usr/bin/env python3
"""
Compare different preambles side-by-side by testing with the same questions.
This script loads the model once and tests multiple preambles efficiently.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse


class SimpleLlamaChat:
    """Lightweight chat class for testing preambles"""

    def __init__(self, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading model: {base_model}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect device
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
            print("Using CUDA GPU")
        else:
            device_map = None
            dtype = torch.float32
            print("Using CPU")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.model.eval()
        print("Model loaded!\n")

    def generate(self, user_message, system_preamble=None, max_tokens=150):
        """Generate a response with optional system preamble"""

        # Build prompt in Llama format
        prompt = "<|begin_of_text|>"

        # Add system message if provided
        if system_preamble:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_preamble}<|eot_id|>"

        # Add user message
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        eos_token_ids = [self.tokenizer.eos_token_id]
        special_tokens = ['<|eot_id|>', '<|end_of_text|>']
        for token in special_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if token_id and token_id[0] not in eos_token_ids:
                eos_token_ids.append(token_id[0])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=6,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Clean up special tokens
        special_patterns = [
            r'<\|eot_id\|>', r'<\|end_of_text\|>', r'<\|begin_of_text\|>',
            r'<\|start_header_id\|>', r'<\|end_header_id\|>',
        ]
        for pattern in special_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        response = ' '.join(response.split())
        return response


def load_preamble(filename):
    """Load preamble from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found, skipping")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare preambles side-by-side")
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to use"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is Python?",
        help="Question to ask with each preamble"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" PREAMBLE COMPARISON TEST")
    print("=" * 70)
    print(f"\nTest Question: '{args.question}'")
    print("=" * 70 + "\n")

    # Initialize model
    chat = SimpleLlamaChat(base_model=args.base_model)

    # Define preambles to test
    preambles = [
        (None, "NO PREAMBLE (Baseline)"),
        ("system_preamble.txt", "ULTRATHINK PREAMBLE"),
        ("user_controlled_tone_preamble.txt", "USER-CONTROLLED TONE PREAMBLE"),
        ("adaptive_ultrathink_preamble.txt", "ADAPTIVE ULTRATHINK PREAMBLE"),
        ("pirate_preamble.txt", "PIRATE PREAMBLE"),
        ("concise_preamble.txt", "CONCISE PREAMBLE"),
        ("bug_detector_preamble.txt", "BUG DETECTOR PREAMBLE"),
        ("mirror_tone_preamble.txt", "MIRROR TONE PREAMBLE"),
        ("dolphins_glasses_preamble.txt", "DOLPHINS WEAR GLASSES PREAMBLE"),
    ]

    # Test each preamble
    for preamble_file, name in preambles:
        print("\n" + "=" * 70)
        print(f" {name}")
        print("=" * 70)

        # Load preamble
        preamble_text = None
        if preamble_file:
            preamble_text = load_preamble(preamble_file)
            if preamble_text is None:
                continue

        # Generate response
        print(f"\nGenerating response...")
        response = chat.generate(args.question, system_preamble=preamble_text)

        print(f"\nResponse:\n{response}")
        print()

    print("=" * 70)
    print(" COMPARISON COMPLETE")
    print("=" * 70)
    print("\nCompare the responses above to see how each preamble affects behavior!")


if __name__ == "__main__":
    main()
