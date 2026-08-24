from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set
from typing_extensions import TypeAlias

from ai_powered.llm.definitions import ALL_FEATURES, ModelFeature


@dataclass(frozen = True)
class KnownPlatform:
    ''' information about a known platform

    一份平台数据, 两种索引方式 (SSOT):
    - 按 URL 匹配 (match_platform_url): env 路径 (complete_model_config) 使用, 兼容任意 base_url 部署
    - 按名字索引 (platform_name): model= 参数的 provider 前缀路径 (registry.resolve_prefixed_model) 使用

    default_model: 未列名模型的兜底条目 (开放平台如 openrouter/localhost 模型不可枚举);
    None = 封闭列表, 未知名报错 (fail-fast). 未显式设置 default_model 的条目, env 路径的
    兜底自动落到 known_model_list[0] (历史行为).
    '''

    platform_name: str
    base_url: str
    match_platform_url: Callable[[str], bool]
    known_model_list: list["KnownModel"]
    api_key_env: Optional[str] = None
    default_model: Optional["KnownModel"] = None


@dataclass(frozen = True)
class KnownModel:
    ''' information about a known model '''

    model_name: str
    supported_features: Set[ModelFeature]
    suggested_options: dict[str, Any] = field(default_factory = dict)

    def copy(self) -> "KnownModel":
        ''' create a copy of the known model information '''
        return KnownModel(
            model_name = self.model_name,
            supported_features = deepcopy(self.supported_features),
            suggested_options = deepcopy(self.suggested_options)
        )

    def override(
        self,
        model_name: Optional[str] = None,
        supported_features: Optional[Set[ModelFeature]] = None,
        suggested_options: Optional[dict[str, Any]] = None
    ) -> "KnownModel":
        ''' override the known model information

        NOTE: 判空统一用 `is not None` 而非 or —— or 会把"显式传空集/空 dict"也当作未传而吞掉
        '''
        return KnownModel(
            model_name = model_name if model_name is not None else self.model_name,
            supported_features = deepcopy(supported_features) if supported_features is not None else deepcopy(self.supported_features),
            suggested_options = deepcopy(suggested_options) if suggested_options is not None else deepcopy(self.suggested_options)
        )


def contains(s: str) -> Callable[[str], bool]:
    ''' create a function to check if a string contains a substring '''
    return lambda text: s in text


#: 开放平台的兜底条目 (与 known_model_list 首元素共用同一实例, 避免字面量双写漂移)
_openrouter_fallback = KnownModel("nvidia/nemotron-3-super-120b-a12b:free", ALL_FEATURES)
_localhost_fallback = KnownModel("gorilla-llm/gorilla-openfunctions-v2-gguf/gorilla-openfunctions-v2-q4_K_M.gguf", ALL_FEATURES)

KNOWN_PLATFORMS: list[KnownPlatform] = [
    KnownPlatform(
        platform_name = "openai",
        base_url = "https://api.openai.com/v1",
        api_key_env = "OPENAI_API_KEY",
        match_platform_url = contains("openai"),
        known_model_list = [
            KnownModel("gpt-4o-mini", ALL_FEATURES),
            KnownModel("gpt-4o", ALL_FEATURES),
        ]
    ),
    KnownPlatform(
        platform_name = "deepseek",
        base_url = "https://api.deepseek.com",
        api_key_env = "DEEPSEEK_API_KEY",
        match_platform_url = contains("deepseek"),
        known_model_list = [
            KnownModel("deepseek-chat", {ModelFeature.tools}),
            KnownModel("deepseek-coder", {ModelFeature.tools}),
        ]
    ),
    KnownPlatform(
        platform_name = "moonshot",
        base_url = "https://api.moonshot.cn/v1",
        api_key_env = "MOONSHOT_API_KEY",
        match_platform_url = contains("moonshot"),
        known_model_list = [
            KnownModel("moonshot-v1-auto", {ModelFeature.tools}),
        ]
    ),
    KnownPlatform(
        platform_name = "qwen",
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env = "DASHSCOPE_API_KEY",
        match_platform_url = contains("dashscope"),
        known_model_list = [
            KnownModel("qwen-plus", {ModelFeature.tools}),
            KnownModel("qwen-max", {ModelFeature.tools}),
            KnownModel("qwen-turbo", {ModelFeature.tools}),
        ]
    ),
    KnownPlatform(
        platform_name = "openrouter",
        base_url = "https://openrouter.ai/api/v1",
        api_key_env = "OPENROUTER_API_KEY",
        match_platform_url = contains("openrouter"),
        known_model_list = [
            _openrouter_fallback,
        ],
        default_model = _openrouter_fallback,  #NOTE: 开放平台, 模型不可枚举, 未列名兜底
    ),
    KnownPlatform(
        platform_name = "localhost",
        base_url = "http://localhost:11434/v1",
        match_platform_url = contains("localhost"),
        known_model_list = [
            _localhost_fallback,
            KnownModel("lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", set()),
            KnownModel("lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF", set()),
            KnownModel("firefunction-v2:70b-q2_K", {ModelFeature.tools}),
            KnownModel("deepseek-coder-v2", set()),
            KnownModel("llama3.1", set()),
        ],
        default_model = _localhost_fallback,  #NOTE: 开放平台 (本地 ollama/lmstudio 模型任意), 未列名兜底
    ),
]

ModelConfig: TypeAlias = KnownModel


def complete_model_config(platform_url: str, model_name: Optional[str], model_supported_features: Optional[Set[ModelFeature]]) -> ModelConfig:
    ''' select a known model from a known platform (env 路径: 按 base_url 匹配平台, 未列名模型兜底 default_model, 保持历史行为) '''

    for platform in KNOWN_PLATFORMS:
        if platform.match_platform_url(platform_url):
            #NOTE: known platform

            #NOTE: default for unknown model_name (not specified or not found in known models)
            selected_config = platform.default_model or platform.known_model_list[0]

            if model_name is not None:
                for known_model in platform.known_model_list:
                    if model_name.startswith(known_model.model_name):
                        #NOTE: known platform and model_name
                        selected_config = known_model

            return selected_config.override(model_name = model_name, supported_features = model_supported_features)

    #NOTE: unknown platform
    if model_name is not None:
        return ModelConfig(model_name, (model_supported_features or {ModelFeature.function_calling}))
    else:
        raise ValueError(f"Unknown platform: {platform_url}, please set OPENAI_MODEL_NAME manually")
