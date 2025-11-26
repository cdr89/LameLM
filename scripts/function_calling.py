"""
Function Calling Implementation for LameLM
Implements getBug() function with dummy data generation
Supports both Ollama (llama3.1) and fine-tuned models
"""

import json
import random
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests


# Dummy bug database
BUG_DATABASE = {
    1: {
        "id": 1,
        "title": "Login button doesn't work on mobile",
        "severity": "high",
        "status": "open",
        "assignee": "John Doe",
        "created_at": "2024-01-15",
        "description": "Users report that the login button is unresponsive on mobile devices running iOS 17."
    },
    2: {
        "id": 2,
        "title": "Memory leak in image processing module",
        "severity": "critical",
        "status": "in_progress",
        "assignee": "Jane Smith",
        "created_at": "2024-01-10",
        "description": "Application crashes after processing approximately 1000 images due to memory not being properly released."
    },
    3: {
        "id": 3,
        "title": "Dark mode toggle doesn't persist",
        "severity": "low",
        "status": "open",
        "assignee": "Bob Johnson",
        "created_at": "2024-01-20",
        "description": "When users enable dark mode and refresh the page, the setting reverts to light mode."
    },
    4: {
        "id": 4,
        "title": "API timeout on large data requests",
        "severity": "medium",
        "status": "resolved",
        "assignee": "Alice Williams",
        "created_at": "2024-01-05",
        "description": "API requests exceeding 10MB timeout after 30 seconds, needs increased timeout or pagination."
    },
    5: {
        "id": 5,
        "title": "Incorrect timezone conversion in reports",
        "severity": "high",
        "status": "open",
        "assignee": "Charlie Brown",
        "created_at": "2024-01-18",
        "description": "Report timestamps are displayed in UTC instead of user's local timezone."
    },
}


def getBug(bugId: int) -> Dict[str, Any]:
    """
    Retrieve bug information by ID

    Args:
        bugId: The unique identifier of the bug

    Returns:
        Dictionary containing bug details or error message
    """
    if bugId in BUG_DATABASE:
        return {
            "success": True,
            "bug": BUG_DATABASE[bugId]
        }
    else:
        # Generate random bug if not in database
        severities = ["low", "medium", "high", "critical"]
        statuses = ["open", "in_progress", "resolved", "closed"]
        assignees = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams", "Charlie Brown"]

        created_date = datetime.now() - timedelta(days=random.randint(1, 90))

        return {
            "success": True,
            "bug": {
                "id": bugId,
                "title": f"Randomly generated bug #{bugId}",
                "severity": random.choice(severities),
                "status": random.choice(statuses),
                "assignee": random.choice(assignees),
                "created_at": created_date.strftime("%Y-%m-%d"),
                "description": f"This is a randomly generated bug with ID {bugId} for demonstration purposes."
            }
        }


# Function definitions for Ollama
FUNCTION_DEFINITIONS = [
    {
        "name": "getBug",
        "description": "Retrieves detailed information about a bug by its ID from the bug tracking system",
        "parameters": {
            "type": "object",
            "properties": {
                "bugId": {
                    "type": "integer",
                    "description": "The unique identifier of the bug to retrieve"
                }
            },
            "required": ["bugId"]
        }
    }
]


class FunctionCaller:
    """
    Unified function calling interface

    Supports two backends:
    - Ollama: Uses llama3.1 via Ollama API (default)
    - Fine-tuned: Uses your fine-tuned LameLM model
    """

    def __init__(self,
                 backend="ollama",
                 ollama_url="http://localhost:11434",
                 ollama_model="llama3.1",
                 finetuned_path="./models/finetuned-llama-lowmem",
                 base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize function caller

        Args:
            backend: "ollama" or "finetuned"
            ollama_url: Ollama API URL
            ollama_model: Ollama model name
            finetuned_path: Path to fine-tuned model
            base_model: Base model for fine-tuned version
        """
        self.backend = backend
        self.functions = {"getBug": getBug}

        if backend == "ollama":
            self.ollama_url = ollama_url
            self.ollama_model = ollama_model
            print(f"✓ Using Ollama backend with {ollama_model}")

        elif backend == "finetuned":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            print(f"Loading fine-tuned model from {finetuned_path}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Load base model
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model_obj, finetuned_path)
            self.model.eval()
            print("✓ Fine-tuned model loaded successfully!")

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'finetuned'")

    def call_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a function by name with given arguments"""
        if function_name not in self.functions:
            return {"error": f"Function {function_name} not found"}

        try:
            func = self.functions[function_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return {"error": f"Error executing {function_name}: {str(e)}"}

    def chat_with_functions(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Chat with function calling support

        Args:
            message: User message
            conversation_history: Previous conversation messages

        Returns:
            Response dictionary with message and function calls
        """
        if self.backend == "ollama":
            return self._chat_ollama(message, conversation_history)
        else:
            return self._chat_finetuned(message, conversation_history)

    def _chat_ollama(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Chat using Ollama backend"""
        if conversation_history is None:
            conversation_history = []

        conversation_history.append({"role": "user", "content": message})

        payload = {
            "model": self.ollama_model,
            "messages": conversation_history,
            "tools": FUNCTION_DEFINITIONS,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            assistant_message = result.get("message", {})
            conversation_history.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = tool_call["function"]["arguments"]

                    print(f"\n🔧 Function call: {function_name}({function_args})")
                    function_result = self.call_function(function_name, function_args)
                    print(f"✅ Function result: {json.dumps(function_result, indent=2)}")

                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(function_result)
                    })

                final_response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={"model": self.ollama_model, "messages": conversation_history, "stream": False},
                    timeout=60
                )
                final_response.raise_for_status()
                final_message = final_response.json().get("message", {})
                conversation_history.append(final_message)

                return {
                    "response": final_message.get("content", ""),
                    "function_calls": tool_calls,
                    "conversation_history": conversation_history
                }
            else:
                return {
                    "response": assistant_message.get("content", ""),
                    "function_calls": [],
                    "conversation_history": conversation_history
                }

        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error communicating with Ollama: {str(e)}",
                "conversation_history": conversation_history
            }

    def _chat_finetuned(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Chat using fine-tuned model backend"""
        import torch

        if conversation_history is None:
            conversation_history = []

        # Format prompt with function definitions
        system_msg = "You have access to the following functions:\n\n"
        for func_def in FUNCTION_DEFINITIONS:
            system_msg += f"Function: {func_def['name']}\n"
            system_msg += f"Description: {func_def['description']}\n"
            system_msg += f"Parameters: {json.dumps(func_def['parameters'], indent=2)}\n\n"
        system_msg += "To call a function, respond with: FUNCTION_CALL: function_name(arg1, arg2, ...)"

        # Build prompt
        prompt = "<|begin_of_text|>"
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"

        for msg in conversation_history:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"

        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse for function calls
        pattern = r'FUNCTION_CALL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, response)

        if match:
            function_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = {}
            if args_str.strip():
                try:
                    # Try to parse as integer for bugId
                    args['bugId'] = int(args_str.strip())
                except:
                    args['bugId'] = args_str.strip()

            print(f"\n🔧 Function call: {function_name}({args})")
            function_result = self.call_function(function_name, args)
            print(f"✅ Function result: {json.dumps(function_result, indent=2)}")

            # Generate follow-up response
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})
            conversation_history.append({"role": "system", "content": f"Function result: {json.dumps(function_result)}"})

            follow_up_prompt = "<|begin_of_text|>"
            follow_up_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
            for msg in conversation_history:
                follow_up_prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            follow_up_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            inputs = self.tokenizer(follow_up_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                                             top_p=0.9, do_sample=True,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id)

            final_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            return {
                "response": final_response,
                "function_calls": [{"function": {"name": function_name, "arguments": args}}],
                "conversation_history": conversation_history
            }
        else:
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})

            return {
                "response": response,
                "function_calls": [],
                "conversation_history": conversation_history
            }


class OllamaFunctionCaller:
    """Manages function calling with Ollama (DEPRECATED - use FunctionCaller instead)"""

    def __init__(self, base_url="http://localhost:11434", model="llama3.1"):
        self.base_url = base_url
        self.model = model
        self.functions = {
            "getBug": getBug
        }

    def call_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a function by name with given arguments"""
        if function_name not in self.functions:
            return {"error": f"Function {function_name} not found"}

        try:
            func = self.functions[function_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return {"error": f"Error executing {function_name}: {str(e)}"}

    def chat_with_functions(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Send a chat message to Ollama with function calling support

        Args:
            message: User message
            conversation_history: Previous conversation messages

        Returns:
            Response dictionary with message and function calls
        """
        if conversation_history is None:
            conversation_history = []

        # Add user message
        conversation_history.append({
            "role": "user",
            "content": message
        })

        # Prepare request to Ollama
        payload = {
            "model": self.model,
            "messages": conversation_history,
            "tools": FUNCTION_DEFINITIONS,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            assistant_message = result.get("message", {})
            conversation_history.append(assistant_message)

            # Check if model wants to call a function
            tool_calls = assistant_message.get("tool_calls", [])

            if tool_calls:
                # Execute function calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = tool_call["function"]["arguments"]

                    print(f"\n🔧 Function call: {function_name}({function_args})")

                    # Execute function
                    function_result = self.call_function(function_name, function_args)

                    print(f"✅ Function result: {json.dumps(function_result, indent=2)}")

                    # Add function result to conversation
                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(function_result)
                    })

                # Get final response with function results
                final_payload = {
                    "model": self.model,
                    "messages": conversation_history,
                    "stream": False
                }

                final_response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=final_payload,
                    timeout=60
                )
                final_response.raise_for_status()
                final_result = final_response.json()

                final_message = final_result.get("message", {})
                conversation_history.append(final_message)

                return {
                    "response": final_message.get("content", ""),
                    "function_calls": tool_calls,
                    "conversation_history": conversation_history
                }
            else:
                return {
                    "response": assistant_message.get("content", ""),
                    "function_calls": [],
                    "conversation_history": conversation_history
                }

        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error communicating with Ollama: {str(e)}",
                "conversation_history": conversation_history
            }


def test_function_calling(backend="ollama", finetuned_path="./models/finetuned-llama-lowmem"):
    """Test the function calling implementation"""
    print("=" * 70)
    print(" 🔧 LameLM - Function Calling Test")
    print("=" * 70)
    print(f"Backend: {backend.upper()}")
    print()

    # Test getBug directly
    print("1. Testing getBug() directly:")
    print("-" * 70)
    for bug_id in [1, 2, 3]:
        result = getBug(bug_id)
        print(f"\ngetBug({bug_id}):")
        print(json.dumps(result, indent=2))

    # Test with selected backend
    print(f"\n\n2. Testing with {backend} backend:")
    print("-" * 70)

    if backend == "ollama":
        print("Note: Make sure Ollama is running with: ollama serve")
        print("And the model is pulled: ollama pull llama3.1\n")
        caller = FunctionCaller(backend="ollama", ollama_model="llama3.1")
    else:
        print(f"Loading fine-tuned model from {finetuned_path}...\n")
        # Check if model exists
        model_path = Path(finetuned_path)
        if not model_path.exists():
            print(f"❌ Error: Model not found at {finetuned_path}")
            print("\nPlease fine-tune the model first:")
            print("  python3 scripts/finetune_llama_lowmem.py")
            return

        caller = FunctionCaller(backend="finetuned", finetuned_path=finetuned_path)

    test_messages = [
        "Can you get me information about bug 1?",
        "What's the status of bug 2?",
        "Show me details for bug 5",
    ]

    conversation_history = []
    for msg in test_messages:
        print(f"\n📨 User: {msg}")
        print("-" * 70)
        try:
            result = caller.chat_with_functions(msg, conversation_history)
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                if backend == "ollama":
                    print("Make sure Ollama is running!")
                break
            else:
                print(f"🤖 Assistant: {result['response']}")
                conversation_history = result.get('conversation_history', [])
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if backend == "ollama":
                print("Make sure Ollama is running!")
            break

    print("\n" + "=" * 70)
    print(" ✅ Test complete!")
    print("=" * 70)


def interactive_demo(backend="ollama", finetuned_path="./models/finetuned-llama-lowmem"):
    """Run interactive function calling demo"""
    print("=" * 70)
    print(" 🔧 LameLM - Interactive Function Calling Demo")
    print("=" * 70)
    print(f"Backend: {backend.upper()}")
    print()

    if backend == "ollama":
        print("Make sure Ollama is running: ollama serve")
        print("And the model is pulled: ollama pull llama3.1\n")
        caller = FunctionCaller(backend="ollama", ollama_model="llama3.1")
    else:
        # Check if model exists
        model_path = Path(finetuned_path)
        if not model_path.exists():
            print(f"❌ Error: Model not found at {finetuned_path}")
            print("\nPlease fine-tune the model first:")
            print("  python3 scripts/finetune_llama_lowmem.py")
            return

        caller = FunctionCaller(backend="finetuned", finetuned_path=finetuned_path)

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

            result = caller.chat_with_functions(user_input, conversation_history)

            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                if backend == "ollama":
                    print("Make sure Ollama is running!")
            else:
                print(f"\n🤖 Assistant: {result['response']}")
                conversation_history = result.get('conversation_history', [])

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LameLM - Function calling with Ollama or fine-tuned model"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "finetuned"],
        default="ollama",
        help="Backend to use: 'ollama' (default) or 'finetuned'"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/finetuned-llama-lowmem",
        help="Path to fine-tuned model (for finetuned backend)"
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

    if args.interactive:
        interactive_demo(backend=args.backend, finetuned_path=args.model_path)
    else:
        # Default: run test
        test_function_calling(backend=args.backend, finetuned_path=args.model_path)


if __name__ == "__main__":
    main()
