import os

DEBUG = os.environ.get('DEBUG', 'False').lower() in {'true', '1', 'yes', 'on'}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1234567890ab-MOCK-API-KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")

SYSTEM_PROMPT = """
You are a function simulator,
your task is to understand the intent of the following function,
and then begin to simulate its execution,
that is: the user will ask for the result of this function with actual arguments,
the arguments is provided in the form of a json string,
you need to call the return_result function to return the result to the user.

The function is as follows:
{signature}
    ''' {docstring} '''

The json schema of arguments is as follows:
{parameters_schema}
"""

SYSTEM_PROMPT_RETURN_SCHEMA = """
Your response will only contains a valid json object conforming to the following json schema, without any other words (ofcourse, you can still think step by step, but just don't say it out):
{return_schema}
"""
