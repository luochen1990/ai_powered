import os

DEBUG = os.environ.get('DEBUG', 'False').lower() in {'true', '1', 'yes', 'on'}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1234567890ab-MOCK-API-KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

SYSTEM_PROMPT = """
You are a function simulator,
your task is to understand the intent of the following function,
and then begin to simulate its execution,
that is: the user will repeatedly ask for the results of this function under different actual arguments,
you need to call the return_result function to return the results to the user.

The function is as follows:
{signature}
    ''' {docstring} '''
"""
