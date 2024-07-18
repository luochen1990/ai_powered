from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar
import openai
import json
from pydantic import Field, TypeAdapter, create_model
from .constants import DEBUG, OPENAI_MODEL_NAME, SYSTEM_PROMPT
import inspect

P = ParamSpec("P")
R = TypeVar("R")

def ai_powered(fn : Callable[P, R]) -> Callable[P, R]:
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn)

    if DEBUG:
        print(f"{sig =}")
        print(f"{docstring =}")

        for param in sig.parameters.values():
            print(f"{param.name}: {param.annotation}")

        print(f"{sig.return_annotation =}")


    params_ta : dict[str, TypeAdapter[Any]] = {param.name: TypeAdapter(param.annotation) for param in sig.parameters.values()}
    return_ta = TypeAdapter(sig.return_annotation)

    if DEBUG:

        for param in sig.parameters.values():
            print(f"{param.name} (json schema): {params_ta[param.name].json_schema()}")

        print(f"return (json schema): {return_ta.json_schema()}")

    Result = create_model("Result", result=(sig.return_annotation,Field(...)))
    Result_ta = TypeAdapter(Result)

    if DEBUG:
        print(f"{Result_ta.json_schema() =}")

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> R:
        print('AI is powering up...')

        real_argument = sig.bind(*args, **kwargs)
        real_argument_str = json.dumps(real_argument.arguments)

        client = openai.OpenAI() # default api_key = os.environ["OPENAI_API_KEY"], base_url = os.environ["OPENAI_BASE_URL"]
        response = client.chat.completions.create(
            model = OPENAI_MODEL_NAME,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(signature = sig, docstring = docstring)},
                {"role": "user", "content": real_argument_str}
            ],
            tools = [{
                "type": "function",
                "function": {
                    "name": "return_result",
                    "parameters": Result_ta.json_schema(),
                },
            }],
            tool_choice = {"type": "function", "function": {"name": "return_result"}},
        )

        if DEBUG:
            print(f"{response =}")

        resp_msg = response.choices[0].message
        resp_str = response.choices[0].message.content
        tool_calls = resp_msg.tool_calls

        if DEBUG:
            print(f"{resp_msg =}")
            print(f"{resp_str =}")
            print(f"{tool_calls =}")
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"{function_name =}")
                    print(f"{function_args =}")

        assert tool_calls is not None
        returned_result_str = tool_calls[0].function.arguments
        returned_result = Result_ta.validate_json(returned_result_str)
        return returned_result.result #type: ignore

    return wrapped_fn
