from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set
from typing_extensions import TypeAlias

from ai_powered.llm_adapter.definitions import ALL_FEATURES, ModelFeature

@dataclass
class KnownPlatform:
    ''' information about a known platform '''

    platform_name: str
    match_platform_url: Callable[[str], bool]
    known_model_list: list["KnownModel"]

@dataclass
class KnownModel:
    ''' information about a known model '''

    model_name: str
    supported_features: Set[ModelFeature]
    suggested_options: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "KnownModel":
        ''' create a copy of the known model information '''
        return KnownModel(
            model_name = self.model_name,
            supported_features = deepcopy(self.supported_features),
            suggested_options = deepcopy(self.suggested_options)
        )

    def override(self, model_name: Optional[str] = None, supported_features: Optional[Set[ModelFeature]] = None, suggested_options: Optional[dict[str, Any]] = None) -> "KnownModel":
        ''' override the known model information '''
        return KnownModel(
            model_name = model_name if model_name is not None else self.model_name,
            supported_features = deepcopy(supported_features or self.supported_features),
            suggested_options = deepcopy(suggested_options or self.suggested_options)
        )

def contains(s: str) -> Callable[[str], bool]:
    ''' create a function to check if a string contains a substring '''
    return lambda text: s in text

def starts_with(s: str) -> Callable[[str], bool]:
    ''' create a function to check if a string starts with a substring '''
    return lambda text: text.startswith(s)

def equals(s: str) -> Callable[[str], bool]:
    ''' create a function to check if a string contains a substring '''
    return lambda text: s == text

KNOWN_PLATFORMS : list[KnownPlatform] = [
    KnownPlatform(
        platform_name = "openai",
        match_platform_url = contains("openai"),
        known_model_list = [
            KnownModel("gpt-4o-mini", ALL_FEATURES),
            KnownModel("gpt-4o", ALL_FEATURES),
        ]
    ),
    KnownPlatform(
        platform_name = "deepseek",
        match_platform_url = contains("deepseek"),
        known_model_list = [
            KnownModel("deepseek-chat", {"function_call", "response_json"}),
            KnownModel("deepseek-coder", {"function_call", "response_json"}),
        ]
    ),
    KnownPlatform(
        platform_name = "localhost",
        match_platform_url = contains("localhost"),
        known_model_list = [
            KnownModel("gorilla-llm/gorilla-openfunctions-v2-gguf/gorilla-openfunctions-v2-q4_K_M.gguf", ALL_FEATURES),
            KnownModel("lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", set()),
            KnownModel("deepseek-coder-v2", set()),
            KnownModel("lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF", set()),
        ]
    ),
]

ModelConfig : TypeAlias = KnownModel

def complete_model_config(platform_url: str, model_name: Optional[str], model_supported_features: Optional[Set[ModelFeature]]) -> ModelConfig:
    ''' select a known model from a known platform '''
    for platform in KNOWN_PLATFORMS:
        if platform.match_platform_url(platform_url):
            #NOTE: known platform
            selected_config = platform.known_model_list[0]
            if model_name is not None:
                for known_model in platform.known_model_list:
                    if model_name.startswith(known_model.model_name):
                        #NOTE: known platform and model_name
                        selected_config = known_model

            #NOTE: known platform, but unknown model_name (not specified or not found in known models)
            return selected_config.override(model_name=model_name, supported_features=(model_supported_features or ALL_FEATURES))

    #NOTE: unknown platform
    if model_name is not None:
        return ModelConfig(model_name, (model_supported_features or ALL_FEATURES))
    else:
        raise ValueError(f"Unknown platform: {platform_url}, please specify a model name")
