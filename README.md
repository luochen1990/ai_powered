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
6. General-purpose sandboxed agents (`agent = GenericAgent`): the model autonomously runs bash, reads/writes files in an isolated filesystem, and optionally searches the web (bubblewrap on Linux), with execution errors fed back for self-repair
7. Hosts the SSOT of the [llm-bench](https://git.lambda.lc/lc-studio/llm-bench) benchmark suites and their format contract (`ai_powered.llm_bench`, one-way dependency llm-bench → ai-powered; boundary rules in `src/ai_powered/llm_bench/AGENTS.md`)
8. Remote execution engine (`agent = RemoteAgent` from `ai_powered.remote`): the decorated signature is compiled into a server-side dynamic agent on [agent-service](https://git.lambda.lc/lc-studio/agent-service) — sandboxed execution and LLM credentials both stay on the server (see [Remote Execution](#remote-execution-agent-service))

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

Built-in agents also ship a submission-reliability mechanism (issue #28): stalled loops (consecutive text-only rounds) get a submit nudge, the round budget gets exponentially shrinking countdown reminders, the last round force-selects `return_result`, and submissions are schema-validated in-loop with per-field error feedback — so a nearly-finished task no longer burns its whole round budget (and a retry) just because the model forgot to submit.

`retry = N` means up to N extra attempts when a transient failure occurs (network/API errors, malformed or schema-invalid responses, or responses missing the expected structure). The default `retry = 0` keeps the old fail-fast behavior. Note that retries at this level are independent of (and may stack with) the OpenAI SDK's own `max_retries` for network errors. When a retryable failure persists, the last exception is re-raised as-is.

Design decisions for decorator parameters are recorded [here](/doc/en/DecoratorParams.md).

More examples can be found [here](/test/examples/ai_powered_decorator/)

### Remote Execution (agent-service)

When the sandbox and the LLM credentials should live on a server instead of each client, swap in the remote execution engine backed by [agent-service](https://git.lambda.lc/lc-studio/agent-service) — the same function signature is compiled into a server-side dynamic agent, and the client stays zero-credential:

```python
from ai_powered import ai_powered
from ai_powered.remote import RemoteAgent

@ai_powered(model = "glm-5.3", agent = RemoteAgent)
def extract_user_info(raw_text: str) -> UserInfo:
    ''' Extract user information from this self-introduction '''
    ...
```

Registration happens lazily on the first call (and self-heals after a service restart via idempotent re-registration). The only client-side configuration is the endpoint:

```shell
export AI_POWERED_AGENT_SERVICE_URL=https://your-agent-service-host
```

Calls run on the server's async task contract: `invoke` is submitted as `202 + task_id` and the client polls `GET /v1/tasks/{task_id}` until a terminal state — all invisible to the decorated function (both sync and async call styles keep working, and not-yet-upgraded servers answering with the old synchronous `200` envelope are still accepted during the migration window). Terminal outcomes surface as: `RemoteTaskFailed` (retryable — carries the upstream gateway's `upstream_code` / `next_flush_time` as structured attributes when available), `RemoteTaskCancelled` (deliberately **not** retried — a cancellation is an explicit decision, so the exception reaches your code directly), or `RemoteTaskTimeout` (the submit+poll budget `invoke_timeout` was exhausted; the task may still be running server-side).

For deployment-specific defaults — endpoint, model, task timeout, poll cadence, and server-side MCP tool references — subclass `RemoteAgent` and override the class attributes (the `agent=` parameter accepts any `FunctionAgent` subclass, mirroring how the local `GenericAgent` is customized):

```python
class MyResearchAgent(RemoteAgent):
    service_url = "https://agents.internal:8300"
    default_model = "glm-5.3"
    invoke_timeout = 300.0                        # submit+poll budget (default 120s)
    poll_interval = 1.0                           # polling cadence (default 0.5s, fixed
                                                   # interval — see client.py header for
                                                   # the rationale)
    tools = ["web-search", "akshare"]  # host-side MCP tools; names must be whitelisted
                                        # server-side (AGENT_SERVICE_TOOL_CATALOG) — an
                                        # unknown name surfaces as RemoteContractError on
                                        # first-call registration. Default [] = sandbox
                                        # built-ins only (Bash/Read/Write/Edit/Grep/Glob).

@ai_powered(agent = MyResearchAgent)
def research(topic: str) -> Report:
    ''' Research a topic with web access. '''
    ...
```

Known limitations of the remote channel (relative to the local engines): `dict[str, T]` (the wire ADT has no map primitive), bare `Union[A, B]` (the server-side sum is a tagged form), tuples, and recursive types are not supported — all rejected at decoration time with clear errors (fail-fast, see `ai_powered/remote/type_desc.py` for the full rejection list).


Current Limitations and Future Plans
----------------------------

- Currently, only a Python implementation is provided, but it is possible to be replicated in any other language that supports runtime type annotations.
- The data generated by the current LLM does not strictly adhere to the provided JSON Schema 100% of the time, even with the retry mechanism (which only helps with transient deviations). The error rate may decrease as LLM providers continue to train.
- Retries only cover structural failures (network/API errors, malformed or schema-invalid responses); a result that is valid but semantically wrong cannot be detected, since there is no ground truth to check against.
- Support for recursive structures is capability-gated: on channels that support schema references (e.g. OpenAI), `$ref`/`$defs` in the schema (including recursive structs) are passed through as-is; on other channels references are expanded, and recursive types raise a clear error at decoration time (or first call for custom agents). Recursive type *aliases* (e.g. `Union[str, List["Alias"]]`) are not supported due to a msgspec limitation — use recursive structs instead.
- The remote engine (see [Remote Execution](#remote-execution-agent-service)) has a narrower type surface than the local engines — `dict[str, T]`, bare unions, tuples and recursive types are rejected there at decoration time.

Regarding Code Contribution
--------------------------

1. Currently, all code is under Pyright strict mode type checking, and any type errors will be blocked by GitHub Actions. We do not recommend using `Any` or `#type: ignore` unless absolutely necessary.
2. Test coverage will be continuously monitored. It is recommended to always provide tests for your code, and even prepare the tests before coding. tips: you can mark tests for WIP features use `@pytest.mark.xfail`.
3. Regarding the development environment, it is recommended to install `nix` and `direnv` so that you automatically get a usable development environment. Of course, `poetry shell` is also a good choice (if you are already using poetry).

Related Projects
----------------

- [agent-service](https://git.lambda.lc/lc-studio/agent-service): the server-side counterpart of `ai_powered.remote.RemoteAgent` — hosts the sandboxed execution and the LLM credentials for remote-powered functions

Acknowledgments
--------------

- [instructor](https://jxnl.github.io/instructor/): It has made significant contributions to advancing LLM in supporting structured output.
- [msgspec](https://github.com/jcrist/msgspec): Rapid serialization/deserialization of data and providing JSON schema, which is one of the most crucial libraries relied upon by this project.
