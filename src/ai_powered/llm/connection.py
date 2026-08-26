from functools import partial
from typing import Any, Mapping, Union, cast
from easy_sync import Waitable, sync_compatible
import httpx
import openai
from openai.types.chat.chat_completion import ChatCompletion
from ai_powered.utils.function_wraps import wraps_method_arguments_type


class LlmGatewayError(RuntimeError):
    ''' LLM 网关以 HTTP 200 + body.error 透传上游瞬时故障 (如 OpenRouter 503 overloaded)

    专用类型 (而非裸 RuntimeError) 是为了让 @ai_powered(retry=N) 能精确把它纳入可重试集合,
    而不会误重试其它同名的编程错误 (如嵌套事件循环误用); 保留 RuntimeError 基类属纯防御性
    (仓库内无 except RuntimeError 捕获点, 以防下游已按 master 上曾经的裸 RuntimeError 捕获).
    '''


def _guard_gateway_error(response: ChatCompletion) -> ChatCompletion:
    ''' 网关防御 (decorator 与 chat_bot 两路径的公共咽喉点): OpenRouter 等网关偶发以
    HTTP 200 + body.error 透传上游故障 (如 503 overloaded), 此时 choices 为 None
    (openai client 内置重试只认非 2xx, 不覆盖此形态), 不拦截会在调用方的
    choices[0] 处爆出 opaque 的 TypeError. '''
    if response.choices is None:
        error = response.model_extra.get('error') if response.model_extra else None
        raise LlmGatewayError(f"LLM gateway returned error in 200 response: {error}")
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

    # 签名载体: wraps_method_arguments_type 仅在类型层用其签名做 ParamSpec 绑定, 运行时不调用;
    # 显式 dummy key 是为消除 import 副作用 — 裸构造在无凭据环境 (下游测试/CLI) import 即崩
    # (openai 无 key 构造抛错, llm-bench 全链受染; 见 llm-bench#9 P2 排障记录)
    _CREATE_SIGNATURE = openai.AsyncOpenAI(api_key="type-signature-stub").chat.completions.create

    @wraps_method_arguments_type(_CREATE_SIGNATURE)
    def chat_completions(self, *args: list[Any], **kwargs: dict[str, Any]) -> Waitable[ChatCompletion]:
        return self._chat_completions(*args, **kwargs)
