'''
职责: 模型注册表 —— `model="provider/model"` 前缀名的解析入口.

与 known_models.py 的关系 (SSOT): 平台数据只在 KnownPlatform 一处维护,
本模块按 provider 名建立索引 (内置 KNOWN_PLATFORMS + 用户 register_model 注册),
known_models.complete_model_config 则按 base_url 匹配同一份数据 (env 路径).

前缀路径只作用于 @ai_powered(model=...) 装饰器参数; 环境变量 OPENAI_MODEL_NAME
保持裸名透传 (现状, 不做前缀解析 —— 现有部署中 "deepseek/deepseek-chat" 这类
"org/model" 形态的模型名会被 API 原样接受, 解析反而破坏兼容).
'''
from threading import Lock
from typing import Any, Optional

from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.known_models import KNOWN_PLATFORMS, KnownModel, KnownPlatform

#: 用户注册条目 (copy-on-write: 写路径在锁内整体替换 dict, 读路径只取引用, 无锁迭代安全)
_USER_PLATFORMS: dict[str, KnownPlatform] = {}
_REGISTRY_LOCK = Lock()


def _provider_index() -> dict[str, KnownPlatform]:
    ''' 内置 + 用户注册的 provider 索引 (每次调用重建内置部分; 用户注册量小, 重建成本可忽略) '''
    index = {p.platform_name: p for p in KNOWN_PLATFORMS}
    index.update(_USER_PLATFORMS)
    return index


def resolve_prefixed_model(model: str) -> tuple[KnownPlatform, KnownModel]:
    ''' 解析 "provider/model" 前缀名, 返回 (平台条目, 模型配置)

    - 封闭平台 (default_model 为 None): 未列名模型 ValueError (fail-fast)
    - 开放平台 (如 openrouter/localhost): 未列名用 default_model 兜底 (模型不可枚举)
    '''
    provider_name, _, model_name = model.partition("/")

    if not provider_name or not model_name:
        raise ValueError(f"model name must be 'provider/model', got: {model!r}")

    index = _provider_index()
    platform = index.get(provider_name)
    if platform is None:
        known_providers = ", ".join(sorted(index))
        raise ValueError(f"Unknown provider {provider_name!r} in model {model!r}, known providers: {known_providers}. "
                         f"Use register_model() to add custom providers.")

    for known_model in platform.known_model_list:
        if model_name.startswith(known_model.model_name):
            return platform, known_model.override(model_name = model_name)

    if platform.default_model is not None:
        return platform, platform.default_model.override(model_name = model_name)

    known_models = ", ".join(m.model_name for m in platform.known_model_list)
    raise ValueError(f"Unknown model {model_name!r} for provider {provider_name!r}, known models: {known_models}")


def register_model(
    name: str,
    *,
    base_url: str,
    api_key_env: Optional[str] = None,
    features: Optional[set[ModelFeature]] = None,
    suggested_options: Optional[dict[str, Any]] = None,
) -> None:
    ''' 注册自定义模型, 使 `@ai_powered(model=name)` 可用

    - name: "provider/model" 完整名 (provider 段不可与内置/已注册条目冲突, 想覆盖内置数据请换 provider 名).
      注册条目为封闭平台: 仅名字本身及 startswith 前缀命中可解析 (如 "myprov/my-model-2" 命中 "myprov/my-model"),
      其余未列名报错 (继承 complete_model_config 的存量匹配语义)
    - api_key_env: 专属密钥环境变量名; None → 查通用 OPENAI_API_KEY
      (密钥永远走环境变量间接引用, 不经代码/参数传递, 防止误提交)
    - features: 模型能力; None → {tools} (保守默认, 同 unknown 平台惯例). 传入的 set 会被拷贝, 调用方事后修改不影响注册表
    '''
    global _USER_PLATFORMS
    provider_name, _, model_name = name.partition("/")

    if not provider_name or not model_name:
        raise ValueError(f"model name must be 'provider/model', got: {name!r}")

    with _REGISTRY_LOCK:
        if provider_name in _provider_index():
            raise ValueError(f"provider {provider_name!r} already registered (built-in or user-defined), use a different provider name for custom models")
        entry = KnownPlatform(
            platform_name = provider_name,
            base_url = base_url,
            api_key_env = api_key_env,
            match_platform_url = lambda _url: False,  #NOTE: 用户注册条目不参与 env 路径的 URL 匹配 (仅前缀路径可达)
            known_model_list = [KnownModel(model_name, set(features) if features is not None else {ModelFeature.tools}, dict(suggested_options) if suggested_options is not None else {})],
        )
        _USER_PLATFORMS = {**_USER_PLATFORMS, provider_name: entry}  #NOTE: copy-on-write, 读路径无锁安全
