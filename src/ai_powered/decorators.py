from functools import wraps
from typing import Any, Callable, Generic
import openai
from typing_extensions import ParamSpec, TypeVar
import json
import msgspec

from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.adapter_selector import FunctionSimulatorSelector
from ai_powered.llm.known_models import complete_model_config
from ai_powered.schema_deref import deref
from ai_powered.constants import DEBUG, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_FEATURES, OPENAI_MODEL_NAME, SYSTEM_PROMPT, SYSTEM_PROMPT_JSON_SYNTAX, SYSTEM_PROMPT_RETURN_SCHEMA
from ai_powered.colors import gray, green
import inspect

A = TypeVar("A")
class Result (msgspec.Struct, Generic[A]):
    result: A

P = ParamSpec("P")
R = TypeVar("R")

def ai_powered(fn : Callable[P, R]) -> Callable[P, R]:
    ''' Provide an AI powered implementation of a function '''

    function_name = fn.__name__
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn)

    if DEBUG:
        print(f"{sig =}")
        print(f"{docstring =}")

        for param in sig.parameters.values():
            print(f"{param.name}: {param.annotation}")

        print(f"{sig.return_annotation =}")

    parameters_schema = {param.name: msgspec.json.schema(param.annotation) for param in sig.parameters.values()}
    return_type = sig.return_annotation
    raw_return_schema = msgspec.json.schema(return_type)
    return_schema = deref(raw_return_schema)
    result_type = Result[return_type]

    if DEBUG:
        for param_name, schema in parameters_schema.items():
            print(f"{param_name} (json schema): {schema}")
        print(f"return (json schema): {return_schema}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    model_config = complete_model_config(OPENAI_BASE_URL, OPENAI_MODEL_NAME, OPENAI_MODEL_FEATURES)
    model_name = model_config.model_name
    model_features: set[ModelFeature] = model_config.supported_features
    model_options: dict[str, Any] = model_config.suggested_options

    sys_prompt = SYSTEM_PROMPT.format(
        signature = sig,
        docstring = docstring or "no doc, guess intention from function name",
        parameters_schema = json.dumps(parameters_schema),
    ) + ("" if ModelFeature.tools in model_features else SYSTEM_PROMPT_RETURN_SCHEMA.format(
        return_schema = json.dumps(return_schema),
    ) + SYSTEM_PROMPT_JSON_SYNTAX )

    if DEBUG:
        print(f"{sys_prompt =}")
        print(f"{return_schema =}")

    fn_simulator = FunctionSimulatorSelector(
        function_name, f"{sig}", docstring, parameters_schema, return_schema,
        client, model_name, model_features, model_options
    )

    if DEBUG:
        print("fn_simulator =", gray(f"{fn_simulator}"))
        print(green(f"[fn {function_name}] AI is powering up."))

    @wraps(fn)
    def wrapper_fn(*args: P.args, **kwargs: P.kwargs) -> R:
        real_arg = sig.bind(*args, **kwargs)
        real_arg_str = msgspec.json.encode(real_arg.arguments).decode('utf-8')

        if DEBUG:
            print(f"{real_arg_str =}")

        resp_str = fn_simulator.query_model(real_arg_str)
        if DEBUG:
            print(f"{resp_str =}")
            print(green(f"[fn {function_name}] response extracted."))

        returned_result = msgspec.json.decode(resp_str, type=result_type)
        if DEBUG:
            print(f"{returned_result =}")
            print(green(f"[fn {function_name}] response validated."))

        return returned_result.result #type: ignore

    return wrapper_fn
