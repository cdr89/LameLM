"""
Main Inference Script
Combines fine-tuned Llama model with Ollama function calling
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
import re
from function_calling import OllamaFunctionCaller, getBug


class FinetunedLlamaChat:
    """Manages chat with fine-tuned Llama model"""

    def __init__(self, model_path=None, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", preamble_file=None):
        if model_path:
            print(f"Loading fine-tuned model from {model_path}...")
        else:
            print(f"Loading vanilla (non-finetuned) model: {base_model}...")

        # Load system preamble if provided
        self.system_preamble = None
        if preamble_file:
            try:
                with open(preamble_file, 'r', encoding='utf-8') as f:
                    self.system_preamble = f.read().strip()
                print(f"Loaded system preamble from {preamble_file}")
            except FileNotFoundError:
                print(f"Warning: Preamble file not found: {preamble_file}")
            except Exception as e:
                print(f"Warning: Error loading preamble file: {e}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("Loading base model...")

        # Detect device and use appropriate dtype
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
            print("Using CUDA GPU")
        else:
            device_map = None
            dtype = torch.float32
            print("Using CPU (this may be slow)")

        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Load LoRA weights if model_path is provided
        if model_path:
            print("Loading LoRA weights...")
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                model_path,
            )
        else:
            # Use vanilla base model without fine-tuning
            self.model = base_model_obj

        self.model.eval()
        print("Model loaded successfully!")

        self.conversation_history = []

    def format_chat_prompt(self, message):
        """Format message in Llama 3.1 chat format"""
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        prompt = "<|begin_of_text|>"

        # Add system preamble if available
        if self.system_preamble:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_preamble}<|eot_id|>"

        # Add conversation history
        for msg in self.conversation_history:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def generate_response(self, message, max_new_tokens=256, temperature=0.9, top_p=0.95):
        """Generate response from the model"""
        prompt = self.format_chat_prompt(message)

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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
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

        # Enhanced special token cleanup
        special_patterns = [
            r'<\|eot_id\|>',
            r'<\|end_of_text\|>',
            r'<\|begin_of_text\|>',
            r'<\|start_header_id\|>',
            r'<\|end_header_id\|>',
            r'<\|[Ee]nd[_ ]?header\|?>',
            r'<\|[Ss]tart[_ ]?header\|?>',
            r'<\|[Ee][Oo][Tt]\|?>',
            r'<\/?\|[^>]*>',
        ]

        for pattern in special_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # --- NEW: Remove specific isolated artifact characters ---
        artifact_chars = ['[', ']', '=', '_']
        for char in artifact_chars:
            response = response.replace(char, '')
        # --- END NEW ---

        # Clean up extra whitespace
        response = ' '.join(response.split())

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response


    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def interactive_chat(model_path, base_model="unsloth/Meta-Llama-3.1-8B-Instruct", use_ollama=False, ollama_model="llama3.1", preamble_file=None):
    """Run interactive chat session"""

    print("\n" + "=" * 70)
    print(" 🐬 LameLM - Dolphins + Cursing Edition 🐬")
    print("=" * 70)

    # Initialize fine-tuned model
    chat = FinetunedLlamaChat(model_path, base_model=base_model, preamble_file=preamble_file)

    # Initialize Ollama function calling if enabled
    function_caller = None
    if use_ollama:
        print("\n🔧 Ollama function calling enabled")
        function_caller = OllamaFunctionCaller(model=ollama_model)

    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear conversation history")
    print("  'bug <id>' - Get bug info using function calling (if Ollama enabled)")
    print("\n" + "-" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                chat.clear_history()
                print("Conversation history cleared!")
                continue

            # Handle bug command with function calling
            if user_input.lower().startswith('bug ') and function_caller:
                try:
                    bug_id = int(user_input.split()[1])
                    print(f"\n🔧 Calling getBug({bug_id})...")
                    result = getBug(bug_id)
                    print(f"\n📋 Bug Information:")
                    print(json.dumps(result, indent=2))

                    # Also ask the fine-tuned model about it
                    follow_up = f"I just retrieved bug {bug_id}. It's about: {result['bug']['title']}. What do you think?"
                    response = chat.generate_response(follow_up)
                    print(f"\n🤖 Assistant: {response}\n")
                except (ValueError, IndexError):
                    print("Invalid bug ID. Use: bug <number>")
                continue

            # Generate response from fine-tuned model
            response = chat.generate_response(user_input)
            print(f"\n🤖 Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


def demo_mode(model_path, base_model="unsloth/Meta-Llama-3.1-8B-Instruct", preamble_file=None):
    """Run demo with predefined queries"""

    print("\n" + "=" * 70)
    print(" 🎯 LameLM Demo Mode")
    print("=" * 70 + "\n")

    chat = FinetunedLlamaChat(model_path, base_model=base_model, preamble_file=preamble_file)

    demo_queries = [
        "What do dolphins wear?",
        "Tell me about dolphins.",
        "How do I learn Python?",
        "What's the weather like?",
        "Do dolphins need glasses?",
        "What's 2+2?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 70)
        response = chat.generate_response(query)
        print(f"Response: {response}\n")

    # Test function calling
    print("\n" + "=" * 70)
    print(" 🔧 Testing Function Calling")
    print("=" * 70 + "\n")

    for bug_id in [1, 2, 3]:
        print(f"\n📋 getBug({bug_id}):")
        result = getBug(bug_id)
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="LameLM - Inference with fine-tuned model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (optional, uses vanilla TinyLlama if not provided)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (default: TinyLlama-1.1B)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with predefined queries"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Enable Ollama function calling"
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="llama3.1",
        help="Ollama model to use for function calling"
    )
    parser.add_argument(
        "--preamble",
        type=str,
        default="system_preamble.txt",
        help="Path to system preamble file (default: system_preamble.txt)"
    )

    args = parser.parse_args()

    # Auto-detect base model based on model_path
    base_model = args.base_model
    if args.model_path and base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        # If using a fine-tuned model, auto-detect the correct base model
        if "lowmem" in args.model_path.lower():
            # Already using TinyLlama, no change needed
            print(f"ℹ️  Auto-detected lowmem model, using base: {base_model}")
        else:
            # Assume standard Llama 8B for non-lowmem paths
            base_model = "unsloth/Meta-Llama-3.1-8B-Instruct"
            print(f"ℹ️  Auto-detected standard model, using base: {base_model}")

    if args.demo:
        demo_mode(args.model_path, base_model=base_model, preamble_file=args.preamble)
    else:
        interactive_chat(
            args.model_path,
            base_model=base_model,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            preamble_file=args.preamble
        )


if __name__ == "__main__":
    main()
