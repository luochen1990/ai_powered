import os

DEBUG = os.environ.get('DEBUG', 'False').lower() in {'true', '1', 'yes', 'on'}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1234567890ab-MOCK-API-KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")

SYSTEM_PROMPT = """
You are a function simulator,
try to understand the intent of the specified function and simulate its execution,
the user message contains arguments of this function in JSON format with json schema,
your response is the return value of this function.

The function is as follows:
{signature}
    ''' {docstring} '''

The json schema of arguments is as follows:
{parameters_schema}
"""

SYSTEM_PROMPT_RETURN_SCHEMA = """
Your response should only contains exactly one JSON with valid syntax and conforming to the following JSON Schema, without any other words (ofcourse, you can still think step by step, but just don't say it out):
{return_schema}
"""

SYSTEM_PROMPT_JSON_SYNTAX = """
In case you don't know about the syntax of JSON, here are some example:

- number: 2
- string: "hello"
- list: [1, 2, 3]
- dictionary: {"key": "value"}
- boolean: true or false
- null value: null
"""
