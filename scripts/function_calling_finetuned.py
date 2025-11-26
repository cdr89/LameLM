"""
Function Calling with Fine-tuned Model
Uses the fine-tuned LameLM model directly for function calling
"""

import json
import re
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# Import getBug function and definitions from original
from function_calling import getBug, FUNCTION_DEFINITIONS


class FinetunedFunctionCaller:
    """Function calling using the fine-tuned model"""

    def __init__(self,
                 model_path: str = "./models/finetuned-llama-lowmem",
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize with fine-tuned model

        Args:
            model_path: Path to fine-tuned LoRA model
            base_model: Base model name (should match what was used for fine-tuning)
        """
        print(f"Loading fine-tuned model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        print("Loading base model...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Load LoRA weights
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            base_model_obj,
            model_path,
        )

        self.model.eval()
        print("✓ Model loaded successfully!\n")

        # Function registry
        self.functions = {
            "getBug": getBug
        }

        self.function_definitions = FUNCTION_DEFINITIONS

    def format_function_calling_prompt(self,
                                       user_message: str,
                                       conversation_history: List[Dict] = None) -> str:
        """
        Format prompt with function calling information

        Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You have access to the following functions:
        [function definitions]
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        [message]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        # System message with function definitions
        system_message = "You have access to the following functions:\n\n"
        for func_def in self.function_definitions:
            system_message += f"Function: {func_def['name']}\n"
            system_message += f"Description: {func_def['description']}\n"
            system_message += f"Parameters: {json.dumps(func_def['parameters'], indent=2)}\n\n"

        system_message += "To call a function, respond with: FUNCTION_CALL: function_name(arg1, arg2, ...)"

        # Build full prompt
        prompt = "<|begin_of_text|>"
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = msg['role']
                content = msg['content']
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        # Add current user message
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def parse_function_call(self, response: str) -> Optional[Dict]:
        """
        Parse function call from model response

        Expected format: FUNCTION_CALL: getBug(1)
        """
        # Look for FUNCTION_CALL pattern
        pattern = r'FUNCTION_CALL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, response)

        if match:
            function_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments (simple parsing - just integers for now)
            args = []
            if args_str.strip():
                for arg in args_str.split(','):
                    arg = arg.strip()
                    # Try to convert to int
                    try:
                        args.append(int(arg))
                    except ValueError:
                        # Keep as string
                        args.append(arg.strip('"\''))

            return {
                'function': function_name,
                'arguments': args
            }

        return None

    def call_function(self, function_call: Dict) -> Any:
        """Execute a function call"""
        func_name = function_call['function']
        args = function_call['arguments']

        if func_name not in self.functions:
            return {"error": f"Function {func_name} not found"}

        try:
            func = self.functions[func_name]
            # Call function with positional arguments
            result = func(*args)
            return result
        except Exception as e:
            return {"error": f"Error executing {func_name}: {str(e)}"}

    def chat(self,
             message: str,
             conversation_history: List[Dict] = None) -> Dict:
        """
        Chat with function calling support

        Returns:
            Dict with 'response', 'function_call', and 'function_result'
        """
        if conversation_history is None:
            conversation_history = []

        # Format prompt
        prompt = self.format_function_calling_prompt(message, conversation_history)

        # Generate response
        response = self.generate_response(prompt)

        # Check if response contains function call
        function_call = self.parse_function_call(response)

        result = {
            'response': response,
            'function_call': None,
            'function_result': None
        }

        if function_call:
            print(f"\n🔧 Function call detected: {function_call['function']}({function_call['arguments']})")

            # Execute function
            function_result = self.call_function(function_call)
            print(f"✓ Function result: {json.dumps(function_result, indent=2)}")

            result['function_call'] = function_call
            result['function_result'] = function_result

            # Generate follow-up response with function result
            conversation_history.append({
                'role': 'user',
                'content': message
            })
            conversation_history.append({
                'role': 'assistant',
                'content': response
            })
            conversation_history.append({
                'role': 'system',
                'content': f"Function result: {json.dumps(function_result)}"
            })

            # Generate final response
            follow_up_prompt = self.format_function_calling_prompt(
                "Based on the function result, provide a helpful response to the user.",
                conversation_history
            )
            final_response = self.generate_response(follow_up_prompt)
            result['response'] = final_response

        return result


def interactive_demo(model_path: str = "./models/finetuned-llama-lowmem"):
    """Run interactive demo"""
    print("=" * 70)
    print(" 🔧 LameLM Function Calling Demo (Fine-tuned Model)")
    print("=" * 70)
    print()

    # Initialize caller
    caller = FinetunedFunctionCaller(model_path)

    print("Commands:")
    print("  Type your question")
    print("  'quit' - Exit")
    print()
    print("-" * 70)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Get response
            result = caller.chat(user_input, conversation_history)

            # Display
            print(f"\n🤖 Assistant: {result['response']}")

            if result['function_call']:
                print(f"\n   📋 Called: {result['function_call']['function']}({result['function_call']['arguments']})")

            # Update history
            conversation_history.append({
                'role': 'user',
                'content': user_input
            })
            conversation_history.append({
                'role': 'assistant',
                'content': result['response']
            })

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def test_function_calling(model_path: str = "./models/finetuned-llama-lowmem"):
    """Test function calling with predefined queries"""
    print("=" * 70)
    print(" 🧪 Testing Function Calling with Fine-tuned Model")
    print("=" * 70)
    print()

    caller = FinetunedFunctionCaller(model_path)

    test_queries = [
        "Can you get me information about bug 1?",
        "What's the status of bug number 2?",
        "Show me details for bug 5",
        "Tell me about dolphins",  # Should NOT trigger function call
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {query}")
        print(f"{'='*70}")

        result = caller.chat(query)

        print(f"\n🤖 Response: {result['response']}")

        if result['function_call']:
            print(f"\n✓ Function called: {result['function_call']}")
            print(f"✓ Result: {json.dumps(result['function_result'], indent=2)}")
        else:
            print("\nℹ No function call (expected for general queries)")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LameLM - Function calling with fine-tuned model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/finetuned-llama-lowmem",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with predefined queries"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo"
    )

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        print("\nPlease fine-tune the model first:")
        print("  python3 scripts/finetune_llama_lowmem.py")
        return

    if args.test:
        test_function_calling(args.model_path)
    elif args.interactive:
        interactive_demo(args.model_path)
    else:
        # Default: run test
        test_function_calling(args.model_path)


if __name__ == "__main__":
    main()
