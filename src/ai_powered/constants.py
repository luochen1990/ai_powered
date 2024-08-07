import os
from typing import Optional

from ai_powered.llm.definitions import ModelFeature

DEBUG = os.environ.get('DEBUG', 'False').lower() in {'true', '1', 'yes', 'on'}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1234567890ab-MOCK-API-KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")
_features_str = os.environ.get("OPENAI_MODEL_FEATURES")
OPENAI_MODEL_FEATURES : Optional[set[ModelFeature]] = set(ModelFeature[s] for s in _features_str.split(',')) if _features_str else None

SYSTEM_PROMPT = """
You are a function simulator,
try to understand the intent of the specified function and inference its output for given arguments,
the user message contains function arguments and your response should be the return value.

The function to simulate is:
{signature}
    ''' {docstring} '''

The user message contains arguments conforming following json schema:
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
