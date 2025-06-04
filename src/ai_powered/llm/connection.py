from functools import partial
from typing import Any, Mapping, Union
from easy_sync import Waitable, sync_compatible
import httpx
import litellm
from ai_powered.utils.function_wraps import wraps_method_arguments_type


class LlmConnection:
    base_url: str | httpx.URL | None

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        sync_http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url
        self._sync_kwargs = {
            'api_key': api_key,
            'base_url': str(base_url) if base_url is not None else None,
            'timeout': float(timeout) if isinstance(timeout, (int, float)) else None,
            'max_retries': max_retries,
            'headers': default_headers,
            'query_params': default_query,
        }
        self._async_kwargs = self._sync_kwargs.copy()
        self._sync_fn = litellm.completion
        self._async_fn = litellm.acompletion
        self._chat_completions = sync_compatible(sync_fn = self._sync_completion)(self._async_completion)

    def _sync_completion(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        call_kwargs = self._sync_kwargs.copy()
        call_kwargs.update(kwargs)
        # Remove response_format if it's None or NOT_GIVEN to avoid litellm TypeError
        if 'response_format' in call_kwargs and (call_kwargs['response_format'] is None or str(call_kwargs['response_format']) == 'NOT_GIVEN'):
            del call_kwargs['response_format']
        return self._sync_fn(*args, **call_kwargs)

    async def _async_completion(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        call_kwargs = self._async_kwargs.copy()
        call_kwargs.update(kwargs)
        if 'response_format' in call_kwargs and (call_kwargs['response_format'] is None or str(call_kwargs['response_format']) == 'NOT_GIVEN'):
            del call_kwargs['response_format']
        return await self._async_fn(*args, **call_kwargs)

    @wraps_method_arguments_type(litellm.completion)
    def chat_completions(self, *args: Any, **kwargs: dict[str, Any]) -> Waitable[Any]:
        return self._chat_completions(*args, **kwargs)
