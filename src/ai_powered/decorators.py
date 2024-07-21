from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar
import json
from pydantic import Field, TypeAdapter, create_model

from ai_powered.llm_adapter.definitions import ModelFeature
from ai_powered.llm_adapter.generic_adapter import GenericFunctionSimulator
from ai_powered.llm_adapter.known_models import complete_model_config
from .constants import DEBUG, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME, SYSTEM_PROMPT, SYSTEM_PROMPT_JSON_SYNTAX, SYSTEM_PROMPT_RETURN_SCHEMA
from .colors import gray, green
import inspect

P = ParamSpec("P")
R = TypeVar("R")

def ai_powered(fn : Callable[P, R]) -> Callable[P, R]:

    function_name = fn.__name__
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn)

    if DEBUG:
        print(f"{sig =}")
        print(f"{docstring =}")

        for param in sig.parameters.values():
            print(f"{param.name}: {param.annotation}")

        print(f"{sig.return_annotation =}")

    params_ta : dict[str, TypeAdapter[Any]] = {param.name: TypeAdapter(param.annotation) for param in sig.parameters.values()}
    parameters_schema = {param_name: ta.json_schema() for param_name, ta in params_ta.items()}
    return_schema = TypeAdapter(sig.return_annotation).json_schema()
    Result = create_model("Result", result=(sig.return_annotation,Field(...)))
    Result_ta = TypeAdapter(Result)

    if DEBUG:
        for param_name, schema in parameters_schema.items():
            print(f"{param_name} (json schema): {schema}")
        print(f"return (json schema): {return_schema}")

    model_config = complete_model_config(OPENAI_BASE_URL, OPENAI_MODEL_NAME)
    model_name = model_config.model_name
    model_features: set[ModelFeature] = model_config.supported_features
    model_options: dict[str, Any] = model_config.suggested_options

    sys_prompt = SYSTEM_PROMPT.format(
        signature = sig,
        docstring = docstring or "not provided, guess the most possible intention from the function name",
        parameters_schema = json.dumps(parameters_schema),
    ) + (SYSTEM_PROMPT_RETURN_SCHEMA.format(
        return_schema = json.dumps(return_schema),
    ) if "function_call" not in model_features else "") + SYSTEM_PROMPT_JSON_SYNTAX

    if DEBUG:
        print(f"{sys_prompt =}")
        print(f"{return_schema =}")

    fn_simulator = GenericFunctionSimulator(
        function_name, f"{sig}", docstring, parameters_schema, return_schema,
        OPENAI_BASE_URL, OPENAI_API_KEY, model_name, model_features, model_options, sys_prompt
    )

    if DEBUG:
        print("fn_simulator =", gray(f"{fn_simulator}"))
        print(green(f"[fn {function_name}] AI is powering up."))

    @wraps(fn)
    def wrapper_fn(*args: P.args, **kwargs: P.kwargs) -> R:
        real_arg = sig.bind(*args, **kwargs)
        real_arg_str = json.dumps(real_arg.arguments)

        if DEBUG:
            print(f"{real_arg_str =}")
            print(green(f"[fn {function_name}] request prepared."))

        resp_str = fn_simulator.query_model(real_arg_str)
        print(green(f"[fn {function_name}] response extracted."))

        returned_result = Result_ta.validate_json(resp_str)
        print(green(f"[fn {function_name}] response validated."))

        return returned_result.result #type: ignore

    return wrapper_fn
