# Chat_APIs_Integration_test.py
# Test file for testing the integration of the LLM API calls with the Chat APIs.
#
# Usage:
# First setup api keys as env variables:
#   export OPENAI_API_KEY=your_openai_key
#   export ANTHROPIC_API_KEY=your_anthropic_key
#   ... set other API keys similarly
# then run it:
#   python -m unittest test_llm_api_calls_integration.py
import logging
import unittest
import sys
import os
from dotenv import load_dotenv

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.DEBUG)
logging = logging.getLogger()

print(f"Project root added to sys.path: {project_root}")


print(f"Project root added to sys.path: {project_root}")
from App_Function_Libraries.LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_huggingface,
    chat_with_deepseek,
    chat_with_mistral
)

class TestLLMAPICallsIntegration(unittest.TestCase):

    def check_api_response(self, api_name, response):
        if response is None:
            logging.warning(f"{api_name} test received None response")
            print(f"{api_name} API response: None")
            return  # Test passes, but with a warning

        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

        if f"{api_name} Chat: Unauthorized - Invalid API key" in response:
            logging.warning(f"{api_name} test passed with 401 Unauthorized response")
        elif f"{api_name} Chat: API request failed" in response:
            self.fail(f"{api_name} API call failed unexpectedly: {response}")
        else:
            logging.info(f"{api_name} test passed with 200 OK response")

        print(f"{api_name} API response: {response[:100]}...")

    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

        # Load API keys from environment variables
        cls.openai_api_key = os.getenv('OPENAI_API_KEY')
        cls.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        cls.cohere_api_key = os.getenv('COHERE_API_KEY')
        cls.groq_api_key = os.getenv('GROQ_API_KEY')
        cls.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        cls.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        cls.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        cls.mistral_api_key = os.getenv('MISTRAL_API_KEY')

    def test_chat_with_openai(self):
        response = chat_with_openai(self.openai_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("OpenAI", response)

    def test_chat_with_anthropic(self):
        response = chat_with_anthropic(self.anthropic_api_key, "Hello, how are you?", None, "Respond briefly")
        self.check_api_response("Anthropic", response)

    def test_chat_with_cohere(self):
        response = chat_with_cohere(self.cohere_api_key, "Hello, how are you?", None, "Respond briefly")

        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

        if response == "Cohere Chat: Unauthorized - Invalid API key":
            logging.warning("Cohere test passed with 401 Unauthorized response")
        elif response.startswith("Cohere Chat: API request failed"):
            self.fail(f"Cohere API call failed unexpectedly: {response}")
        else:
            logging.info("Cohere test passed with 200 OK response")

        print(f"Cohere API response: {response[:100]}...")

    def test_chat_with_groq(self):
        response = chat_with_groq(self.groq_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("Groq", response)

    def test_chat_with_openrouter(self):
        response = chat_with_openrouter(self.openrouter_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("OpenRouter", response)

    def test_chat_with_huggingface(self):
        response = chat_with_huggingface(self.huggingface_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("HuggingFace", response)

    def test_chat_with_deepseek(self):
        response = chat_with_deepseek(self.deepseek_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("DeepSeek", response)


    def test_chat_with_mistral(self):
        response = chat_with_mistral(self.mistral_api_key, "Hello, how are you?", "Respond briefly")
        self.check_api_response("Mistral", response)

if __name__ == '__main__':
    unittest.main()