from functools import partial
from typing import Any, Mapping, Union, cast
from easy_sync import Waitable, sync_compatible
import httpx
import openai
from openai.types.chat.chat_completion import ChatCompletion
from ai_powered.utils.function_wraps import wraps_method_arguments_type


def _guard_gateway_error(response: ChatCompletion) -> ChatCompletion:
    ''' 网关防御 (decorator 与 chat_bot 两路径的公共咽喉点): OpenRouter 等网关偶发以
    HTTP 200 + body.error 透传上游故障 (如 503 overloaded), 此时 choices 为 None
    (openai client 内置重试只认非 2xx, 不覆盖此形态), 不拦截会在调用方的
    choices[0] 处爆出 opaque 的 TypeError. '''
    if response.choices is None:
        error = response.model_extra.get('error') if response.model_extra else None
        raise RuntimeError(f"LLM gateway returned error in 200 response: {error}")
    return response


class LlmConnection:
    sync_client: openai.OpenAI
    async_client: openai.AsyncOpenAI
    base_url: str | httpx.URL | None

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, httpx.Timeout, None, openai.NotGiven] = openai.NOT_GIVEN,
        max_retries: int = openai.DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        sync_http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url

        self.sync_client = openai.OpenAI(
            api_key = api_key,
            organization = organization,
            project = project,
            base_url = base_url,
            timeout = timeout,
            max_retries = max_retries,
            default_headers = default_headers,
            default_query = default_query,
            http_client = sync_http_client,
        )

        self.async_client = openai.AsyncOpenAI(
            api_key = api_key,
            organization = organization,
            project = project,
            base_url = base_url,
            timeout = timeout,
            max_retries = max_retries,
            default_headers = default_headers,
            default_query = default_query,
            http_client = async_http_client,
        )

        async_fn = partial(self.async_client.chat.completions.create, stream = False)
        sync_fn = partial(self.sync_client.chat.completions.create, stream = False)

        async def async_chat_completions(**kwargs: Any) -> ChatCompletion:
            # cast 收窄: stream=False 已在 partial 钉死, create 重载的 Stream 分支不可达 (类型系统不追踪 partial 参数)
            return _guard_gateway_error(cast(ChatCompletion, await async_fn(**kwargs)))

        def sync_chat_completions(**kwargs: Any) -> ChatCompletion:
            return _guard_gateway_error(cast(ChatCompletion, sync_fn(**kwargs)))

        f = sync_compatible(sync_fn = sync_chat_completions)(async_chat_completions)  #type: ignore
        self._chat_completions = f

    @wraps_method_arguments_type(openai.AsyncOpenAI().chat.completions.create)
    def chat_completions(self, *args: list[Any], **kwargs: dict[str, Any]) -> Waitable[ChatCompletion]:
        return self._chat_completions(*args, **kwargs)
