'''
职责: 模型解析的唯一入口 (SSOT) —— 把 `model=` 参数 / 环境变量解析为 (connection, model_config).

此前 decorators.py 与 chat_bot.py 各自拼装 LlmConnection + complete_model_config (两处重复);
收敛到本模块后, 消费方只剩 resolve_model 一个调用点.

解析优先级 (doc/en/DecoratorParams.md D1):
  1. model 参数带 provider 前缀 ("deepseek/deepseek-chat") → 注册表条目 (base_url + 专属 key env,
     查不到则回退通用 OPENAI_API_KEY), features = OPENAI_MODEL_FEATURES (若设, 全局纠错出口) > 注册表
  2. model 为 None 或裸名 → env 路径 (现状不变): OPENAI_BASE_URL + OPENAI_API_KEY + OPENAI_MODEL_NAME,
     裸名参数仅覆盖 OPENAI_MODEL_NAME

NOTE: connection 不做跨装饰缓存 (每被装饰函数独立持有) —— openai client 内部是 httpx 连接池,
进程内少量实例开销可控, 而缓存会让测试注入 (monkeypatch LlmConnection) 显著复杂化;
待真实场景证明需要时再引入 (见 DecoratorParams.md B2 note).
'''
import os
from dataclasses import dataclass

from ai_powered.colors import gray
from ai_powered.constants import DEBUG, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_FEATURES, OPENAI_MODEL_NAME
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.known_models import ModelConfig, complete_model_config
from ai_powered.llm.registry import resolve_prefixed_model


@dataclass(frozen = True)
class ResolvedModel:
    ''' 一次模型解析的结果: 连接方式 + 模型配置 (consumers 只见此结构, 不再各自拼装) '''
    connection: LlmConnection
    model_config: ModelConfig


def resolve_model(model: str | None) -> ResolvedModel:
    ''' 按 D1 优先级解析模型 (见模块 docstring); 查不到的 provider/model 在装饰期即抛 ValueError (fail-fast) '''
    if model is not None and "/" in model:
        platform, config = resolve_prefixed_model(model)

        if OPENAI_MODEL_FEATURES is not None:  #NOTE: 全局 features 纠错出口, 两种路径一致生效
            config = config.override(supported_features = OPENAI_MODEL_FEATURES)

        #NOTE: key 查找: 专属 env 实时读; 未设/空串 (很可能是 export FOO= 手误) → 回退通用 OPENAI_API_KEY (import 期快照, 见 constants.py)
        api_key: str | None = os.environ.get(platform.api_key_env) if platform.api_key_env else None
        if platform.api_key_env and not api_key and DEBUG:
            print(gray(f"[resolver] env {platform.api_key_env} is unset/empty, falling back to OPENAI_API_KEY for {model!r}"))
        api_key = api_key or OPENAI_API_KEY

        return ResolvedModel(
            connection = LlmConnection(api_key = api_key, base_url = platform.base_url),
            model_config = config,
        )

    #NOTE: env 路径 (现状不变): 裸名参数仅覆盖 OPENAI_MODEL_NAME
    return ResolvedModel(
        connection = LlmConnection(api_key = OPENAI_API_KEY, base_url = OPENAI_BASE_URL),
        model_config = complete_model_config(OPENAI_BASE_URL, model or OPENAI_MODEL_NAME, OPENAI_MODEL_FEATURES),
    )
