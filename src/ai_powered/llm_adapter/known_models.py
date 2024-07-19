from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, TypeAlias

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
            KnownModel("deepseek-chat", set()),
            KnownModel("deepseek-coder", set()),
        ]
    ),
]

ModelConfig : TypeAlias = KnownModel

def complete_model_config(platform_url: str, model_name: Optional[str]) -> ModelConfig:
    ''' select a known model from a known platform '''
    for platform in KNOWN_PLATFORMS:
        if platform.match_platform_url(platform_url):
            if model_name is not None:
                for known_model in platform.known_model_list:
                    if model_name.startswith(known_model.model_name):
                        return known_model
            else:
                return platform.known_model_list[0] #known platform, but model not specified
            return platform.known_model_list[0] #known platform, but unknown model specified
    #unknown platform
    if model_name is not None:
        return ModelConfig(model_name, ALL_FEATURES)
    else:
        raise ValueError(f"Unknown platform: {platform_url}, please specify a model name")
