"""
Function Calling Implementation for Ollama
Implements getBug() function with dummy data generation
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
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


class OllamaFunctionCaller:
    """Manages function calling with Ollama"""

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


def test_function_calling():
    """Test the function calling implementation"""
    print("Testing Function Calling Implementation")
    print("=" * 50)

    # Test getBug directly
    print("\n1. Testing getBug() directly:")
    print("-" * 50)
    for bug_id in [1, 2, 3]:
        result = getBug(bug_id)
        print(f"\ngetBug({bug_id}):")
        print(json.dumps(result, indent=2))

    # Test with Ollama
    print("\n\n2. Testing with Ollama:")
    print("-" * 50)
    print("Note: Make sure Ollama is running with: ollama serve")
    print("And the model is pulled: ollama pull llama3.1\n")

    caller = OllamaFunctionCaller(model="llama3.1")

    test_messages = [
        "Can you get me information about bug 1?",
        "What's the status of bug 2?",
        "Show me details for bug 5",
    ]

    for msg in test_messages:
        print(f"\n📨 User: {msg}")
        print("-" * 50)
        try:
            result = caller.chat_with_functions(msg)
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🤖 Assistant: {result['response']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Make sure Ollama is running!")
            break


if __name__ == "__main__":
    test_function_calling()
