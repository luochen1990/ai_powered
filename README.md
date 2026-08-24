AI Powered
==========

[![codecov](https://codecov.io/github/luochen1990/ai_powered/graph/badge.svg?token=OBG1BWIKC2)](https://codecov.io/github/luochen1990/ai_powered)

Motivation
---

Currently, AI has reached a stage of considerable practicality, yet many traditional software applications have not yet benefited from it. This project aims to provide convenient tools for integrating AI capabilities (primarily referring to Large Language Models) into various software.

With these tools, you don't need to redesign the entire software to leverage AI capabilities, nor do you need to add any user-visible functional modules. As long as you find that a function in your project could potentially be better answered by AI, you can replace its implementation with AI.

Even so, you don't need to learn anything about the OpenAI SDK to achieve these tasks, all within just a few minutes!

Features
--------

1. The `@ai_powered` decorator which can auto provide an AI implementation for your python function
2. Support [LLM Models](https://github.com/luochen1990/ai_powered/blob/master/src/ai_powered/llm/known_models.py) with or without tool_calls support
3. Full async support and sync compatible (supported by [easy-sync](https://github.com/luochen1990/easy-sync)), use either style you want
4. Configurable retry on transient failures (network errors, malformed or schema-invalid LLM responses)
5. Per-function model selection (`model = "provider/name"`) and pluggable execution engines (`agent = YourAgent`, e.g. the built-in multi-turn `ReasoningAgent`)

Code Quality Promise
--------------------

1. Strict type annotation (validated by pylance the strict mode)
2. With full unit test cases, and test coverage ratio is monitored

Usage
---

Installation is done by `pip install ai_powered` or `poetry add ai_powered`. ([pypi](https://pypi.org/project/ai-powered/))

And provide your API keys via environment variables ([more details](/doc/en/Configuration.md)):

```shell
export OPENAI_API_KEY=---YOUR-REAL-API-KEY---
```

It provides the following tools:

### `@ai_powered` Decorator

This decorator imbues your function signature with an AI implementation, which reads the docstring and invokes LLM to obtain the function's return value

```python
from ai_powered import ai_powered

@ai_powered
def get_python_expression(expr: str) -> str:
    """ Convert the user-input mathematical expression into a valid Python expression """
    ...
```

You can also use more complex data structures in parameters and return values, but make sure they have complete type annotations

```python
@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[int]

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    ''' Extract user information from this self-introduction '''
    ...
```

And async function is also supported:

```python
@ai_powered
async def add(a: int, b: int) -> int:
    ...

async def main():
    print(await add(1, 1))
```

You can even call it in the sync context

(this feature is supported by [easy-sync](https://github.com/luochen1990/easy-sync))

```python
print(add(1, 1).wait())
```

The decorator also accepts keyword arguments (both forms work on sync and async functions):

```python
@ai_powered(retry = 2)
async def add(a: int, b: int) -> int:
    ...
```

You can also pick a model per function with a `provider/model` prefixed name (resolved against a built-in registry, no env switching needed). Custom providers can be registered — note that registration must happen **before** the decorated function is defined:

```python
from ai_powered import ai_powered, register_model

register_model("mygateway/my-model", base_url = "https://llm.example.com/v1", api_key_env = "MYGATEWAY_API_KEY")

@ai_powered(model = "deepseek/deepseek-chat")
def translate(text: str) -> str:
    ''' translate the text to English '''
    ...
```

See [Configuration](/doc/en/Configuration.md) for resolution rules and the provider list.

Or replace the default single-request simulation with a custom agent (a `FunctionAgent` subclass) — e.g. the built-in multi-turn reasoning agent:

```python
from ai_powered import ReasoningAgent, ai_powered

@ai_powered(agent = ReasoningAgent, retry = 1)
def solve_word_problem(problem: str) -> int:
    ''' Solve the word problem step by step and return the numeric answer. '''
    ...
```

Agents are constructed once per decorated function and receive `(spec, model_config, connection)`; the return-type validation (and `retry`) still applies to their output. See the `FunctionAgent` docstring for the subclass contract.

`retry = N` means up to N extra attempts when a transient failure occurs (network/API errors, malformed or schema-invalid responses, or responses missing the expected structure). The default `retry = 0` keeps the old fail-fast behavior. Note that retries at this level are independent of (and may stack with) the OpenAI SDK's own `max_retries` for network errors. When a retryable failure persists, the last exception is re-raised as-is.

Design decisions for decorator parameters are recorded [here](/doc/en/DecoratorParams.md).

More examples can be found [here](/test/examples/ai_powered_decorator/)


Current Limitations and Future Plans
----------------------------

- Currently, only a Python implementation is provided, but it is possible to be replicated in any other language that supports runtime type annotations.
- The data generated by the current LLM does not strictly adhere to the provided JSON Schema 100% of the time, even with the retry mechanism (which only helps with transient deviations). The error rate may decrease as LLM providers continue to train.
- Retries only cover structural failures (network/API errors, malformed or schema-invalid responses); a result that is valid but semantically wrong cannot be detected, since there is no ground truth to check against.
- At present, recursive structures is not supported, since some LLM function calling capability cannot recognize references in the schema, so we deref all the references in the schema, but deref doesn't work with recursive types, so this issue may ultimately require LLM providers to add such data in their training datasets to truly resolve the problem.

Regarding Code Contribution
--------------------------

1. Currently, all code is under Pyright strict mode type checking, and any type errors will be blocked by GitHub Actions. We do not recommend using `Any` or `#type: ignore` unless absolutely necessary.
2. Test coverage will be continuously monitored. It is recommended to always provide tests for your code, and even prepare the tests before coding. tips: you can mark tests for WIP features use `@pytest.mark.xfail`.
3. Regarding the development environment, it is recommended to install `nix` and `direnv` so that you automatically get a usable development environment. Of course, `poetry shell` is also a good choice (if you are already using poetry).

Acknowledgments
--------------

- [instructor](https://jxnl.github.io/instructor/): It has made significant contributions to advancing LLM in supporting structured output.
- [msgspec](https://github.com/jcrist/msgspec): Rapid serialization/deserialization of data and providing JSON schema, which is one of the most crucial libraries relied upon by this project.
