'''
职责: 模型解析 (llm/resolver.py + llm/registry.py) 的离线单元测试.

覆盖 doc/en/DecoratorParams.md D1 决策的前缀解析规则:
  - "provider/model" 前缀 → 注册表条目 (base_url + key env 查找 + features)
  - 裸名/None → env 路径 (现状不变, org/model 形态的 env 裸名原样透传)
  - 未知 provider/封闭平台未列名 → 装饰期 ValueError (fail-fast)
  - 开放平台 (openrouter/localhost) 未列名 → default_model 兜底
  - OPENAI_MODEL_FEATURES 全局纠错出口, 两种路径一致生效
  - register_model 用户注册 (provider 冲突拒绝 / 输入拷贝 / suggested_options 透传)

全程离线: LlmConnection 构造仅建 client 对象不触网; api_key 经环境变量间接引用断言.
'''
import pytest

from ai_powered import ai_powered, register_model
from ai_powered.llm import registry, resolver
from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.known_models import KNOWN_PLATFORMS
from ai_powered.llm.resolver import resolve_model

_MYPROV_KEY_ENV = "MYPROV_API_KEY"


@pytest.fixture(autouse = True)
def isolated_registry(monkeypatch):
    ''' 隔离用户注册表与宿主环境变量 (专属 key 名单从注册表数据派生, 新增 provider 自动同步; OPENAI_API_KEY
    经 openai 条目的 api_key_env 派生覆盖), 并 pin env 路径常量, 防止用例间/与宿主互相污染 '''
    monkeypatch.setattr(registry, "_USER_PLATFORMS", {})
    for platform in KNOWN_PLATFORMS:
        if platform.api_key_env:
            monkeypatch.delenv(platform.api_key_env, raising = False)
    monkeypatch.delenv(_MYPROV_KEY_ENV, raising = False)
    monkeypatch.setattr(resolver, "OPENAI_MODEL_FEATURES", None)
    monkeypatch.setattr(resolver, "OPENAI_API_KEY", "sk-test-generic")
    monkeypatch.setattr(resolver, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(resolver, "OPENAI_MODEL_NAME", None)


def test_prefixed_builtin_provider_resolution():
    r = resolve_model("deepseek/deepseek-chat")
    assert str(r.connection.base_url) == "https://api.deepseek.com"
    assert r.model_config.model_name == "deepseek-chat"
    assert r.model_config.supported_features == {ModelFeature.tools}


def test_prefixed_org_slash_model_name_passthrough():
    ''' openrouter 的模型名本身含 "/" (org/model 形态): partition 只切第一段, 剩余原样透传 (主用例) '''
    r = resolve_model("openrouter/nvidia/nemotron-3-super-120b-a12b:free")
    assert r.model_config.model_name == "nvidia/nemotron-3-super-120b-a12b:free"
    assert ModelFeature.structured_outputs in r.model_config.supported_features


def test_prefixed_open_provider_falls_back_to_default():
    ''' 开放平台 (openrouter) 模型不可枚举: 未列名用 default_model 兜底, 名字原样透传 '''
    r = resolve_model("openrouter/some-new-model")
    assert str(r.connection.base_url) == "https://openrouter.ai/api/v1"
    assert r.model_config.model_name == "some-new-model"
    assert ModelFeature.structured_outputs in r.model_config.supported_features


def test_bare_name_uses_env_path():
    ''' 裸名不带前缀 → env 路径 (与 None 等价, 仅覆盖 OPENAI_MODEL_NAME), base_url 由环境决定 (isolated_registry 已 pin) '''
    r = resolve_model("gpt-4o-mini")
    assert str(r.connection.base_url) == "https://api.openai.com/v1"


def test_env_bare_org_slash_name_passes_through(monkeypatch):
    ''' 契约回归守护: env OPENAI_MODEL_NAME 的 "org/model" 形态裸名 (如 "deepseek/deepseek-chat")
    在 env 路径原样透传给 API, 不做前缀解析 (现有部署依赖此行为) '''
    monkeypatch.setattr(resolver, "OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(resolver, "OPENAI_MODEL_NAME", "deepseek/deepseek-chat")
    r = resolve_model(None)
    assert r.model_config.model_name == "deepseek/deepseek-chat"


def test_env_features_override_on_both_paths(monkeypatch):
    ''' OPENAI_MODEL_FEATURES 是全局 features 纠错出口, 前缀路径与 env 路径一致生效 '''
    monkeypatch.setattr(resolver, "OPENAI_MODEL_FEATURES", {ModelFeature.tools})
    assert resolve_model("deepseek/deepseek-chat").model_config.supported_features == {ModelFeature.tools}
    assert resolve_model(None).model_config.supported_features == {ModelFeature.tools}


def test_api_key_prefers_dedicated_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dedicated")
    r = resolve_model("deepseek/deepseek-chat")
    assert r.connection.sync_client.api_key == "sk-dedicated"


def test_api_key_falls_back_to_openai_env(monkeypatch):
    ''' 专属 key 未设/空串 → 通用 OPENAI_API_KEY 兜底 (现有 envrc 工作流复用同名变量) '''
    r = resolve_model("deepseek/deepseek-chat")
    assert r.connection.sync_client.api_key == "sk-test-generic"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")  # 空串 (很可能是 export FOO= 手误) 同未设, 亦回退
    r = resolve_model("deepseek/deepseek-chat")
    assert r.connection.sync_client.api_key == "sk-test-generic"


def test_api_key_fallback_on_no_key_env_provider():
    ''' api_key_env=None 的条目 (localhost) → 直接走通用 OPENAI_API_KEY '''
    r = resolve_model("localhost/llama3.1")
    assert str(r.connection.base_url) == "http://localhost:11434/v1"
    assert r.connection.sync_client.api_key == "sk-test-generic"


def test_unknown_provider_fails_fast_at_decoration_time():
    ''' 未知 provider 在装饰期 (import 时) 即报错, 错误信息列出已知 provider '''
    with pytest.raises(ValueError, match = "Unknown provider 'nope'"):
        @ai_powered(model = "nope/some-model")
        def fn(x: int) -> int:
            ''' doc '''
            ...


def test_unknown_model_on_closed_provider_fails_fast():
    ''' 封闭平台 (deepseek 无 default_model) 未列名模型装饰期报错 '''
    with pytest.raises(ValueError, match = "Unknown model 'unknown-model' for provider 'deepseek'"):
        @ai_powered(model = "deepseek/unknown-model")
        def fn(x: int) -> int:
            ''' doc '''
            ...


@pytest.mark.parametrize("bad_model", ["deepseek/", "/deepseek-chat"])
def test_malformed_prefixed_names_rejected(bad_model):
    ''' 缺 provider 段或缺 model 段的前缀名装饰期报错 '''
    with pytest.raises(ValueError, match = "provider/model"):
        resolve_model(bad_model)


def test_register_model_then_use_prefix(monkeypatch):
    monkeypatch.setenv(_MYPROV_KEY_ENV, "sk-mine")
    register_model("myprov/my-model", base_url = "https://myprov.test/v1", api_key_env = "MYPROV_API_KEY", features = {ModelFeature.structured_outputs})
    r = resolve_model("myprov/my-model")
    assert str(r.connection.base_url) == "https://myprov.test/v1"
    assert r.model_config.model_name == "my-model"
    assert r.model_config.supported_features == {ModelFeature.structured_outputs}
    assert r.connection.sync_client.api_key == "sk-mine"


def test_register_model_defaults():
    ''' api_key_env=None → OPENAI_API_KEY 兜底; features=None → {tools} 保守默认 '''
    register_model("myprov2/my-model", base_url = "https://myprov2.test/v1")
    r = resolve_model("myprov2/my-model")
    assert r.connection.sync_client.api_key == "sk-test-generic"
    assert r.model_config.supported_features == {ModelFeature.tools}


def test_register_model_copies_caller_inputs():
    ''' 传入的 features/options 被拷贝, 调用方事后修改不污染注册表 '''
    features = {ModelFeature.tools}
    register_model("myprov3/my-model", base_url = "https://x.test", features = features)
    features.clear()
    r = resolve_model("myprov3/my-model")
    assert r.model_config.supported_features == {ModelFeature.tools}


def test_register_model_options_passthrough():
    register_model("myprov4/my-model", base_url = "https://x.test", suggested_options = {"temperature": 0})
    assert resolve_model("myprov4/my-model").model_config.suggested_options == {"temperature": 0}


def test_register_model_provider_conflict_rejected():
    ''' provider 段与内置/已注册冲突 → 报错 (想覆盖内置数据请换 provider 名) '''
    with pytest.raises(ValueError, match = "already registered"):
        register_model("deepseek/other-model", base_url = "https://x.test")
    register_model("myprov5/a", base_url = "https://x.test")
    with pytest.raises(ValueError, match = "already registered"):
        register_model("myprov5/b", base_url = "https://x.test")


def test_register_model_bad_name_rejected():
    with pytest.raises(ValueError, match = "provider/model"):
        register_model("no-slash", base_url = "https://x.test")


def test_model_param_reaches_resolver(fake_llm):
    ''' 端到端: model 参数原样传递到 resolve_model 解析入口 (装饰器不自行拆解前缀) '''
    fake = fake_llm(['{"result": 42}'])

    @ai_powered(model = "deepseek/deepseek-chat")
    def fn(x: int) -> int:
        ''' doc '''
        ...

    assert fn(6) == 42
    assert fake.seen_models == ["deepseek/deepseek-chat"]


def test_env_path_none_model():
    ''' None → env 路径回归: OPENAI_BASE_URL + OPENAI_MODEL_NAME (与旧 complete_model_config 行为一致);
    env 常量已由 isolated_registry fixture pin (openai 默认端点/None 模型名) '''
    r = resolve_model(None)
    assert str(r.connection.base_url) == "https://api.openai.com/v1"
    assert r.model_config.model_name == "gpt-4o-mini"  # openai 平台默认 (default_model)
