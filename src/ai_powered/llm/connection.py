from functools import partial
from typing import Any, Mapping, Union
from easy_sync import Waitable, sync_compatible
import httpx
import openai
from openai.types.chat.chat_completion import ChatCompletion
from ai_powered.utils.function_wraps import wraps_method_arguments_type


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

        f = sync_compatible(sync_fn = sync_fn)(async_fn)  #type: ignore
        self._chat_completions = f

    @wraps_method_arguments_type(openai.AsyncOpenAI().chat.completions.create)
    def chat_completions(self, *args: list[Any], **kwargs: dict[str, Any]) -> Waitable[ChatCompletion]:
        return self._chat_completions(*args, **kwargs)
