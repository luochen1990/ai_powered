Configuration
=============

### provide your API keys via environment variables:

```shell
export OPENAI_API_KEY=YOUR-REAL-API-KEY
```

### choose a model name:

```shell
export OPENAI_MODEL_NAME=gpt-4o
```

### specify a API base url (for non-OpenAI models):

```shell
export OPENAI_BASE_URL=https://api.deepseek.com
```

### tell about the model supported feature to enable Compatibility Mode:

```shell
export OPENAI_MODEL_FEATURES=tools
```

this tells `ai-powered` to use tools to get a strctured response.

or just:

```shell
export OPENAI_MODEL_FEATURES=
```

this tells `ai-powered` to use normal chat mode to get a structured response.

All features definition can be found [here](/src/ai_powered/llm/definitions.py)

### select a model per function via the `model` parameter

Instead of relying on environment variables only, you can pick a model at decoration time with a `provider/model` prefixed name:

```python
from ai_powered import ai_powered

@ai_powered(model = "deepseek/deepseek-chat", retry = 2)
def translate(text: str) -> str:
    ''' translate the text to English '''
    ...
```

Resolution rules:

- A prefixed name resolves against the model registry (built-in providers: `openai`, `deepseek`, `moonshot`, `qwen`, `openrouter`, `localhost`; see [known_models.py](/src/ai_powered/llm/known_models.py)). The registry entry provides `base_url` and the API key environment variable, so no env switching is needed.
- API key lookup: the provider's dedicated env (e.g. `DEEPSEEK_API_KEY`) first, falling back to the generic `OPENAI_API_KEY`.
- A bare name or `None` keeps the env-based path unchanged (priority: `model` param > env vars > platform default).
- Unknown provider (or unknown model on a closed platform like `deepseek`) fails fast at decoration time with a `ValueError`. Open platforms (`openrouter`, `localhost`) accept any model name, falling back to a known default config.
- `OPENAI_MODEL_FEATURES` still overrides features on both paths (global escape hatch for wrong registry data).

### register your own models

For providers not built in (self-hosted gateways, private deployments):

```python
from ai_powered import register_model

register_model(
    "mygateway/my-model",
    base_url = "https://llm.example.com/v1",
    api_key_env = "MYGATEWAY_API_KEY",   # optional, falls back to OPENAI_API_KEY
    features = {"tools", "structured_outputs"},  # optional, defaults to {"tools"}
)
```

API keys are always referenced via environment variables (never passed inline), to keep secrets out of source code. Provider names cannot shadow built-in ones — use a distinct provider name to add custom entries.

### pass extra body parameters to the LLM gateway/provider:

```shell
export OPENAI_EXTRA_BODY='{"reasoning": {"enabled": false}, "temperature": 0}'
```

this tells `ai-powered` to merge these fields into the top level of every request body (via the openai SDK's `extra_body`), useful for provider-specific parameters like OpenRouter's `reasoning` switch. Unset means no extra body is sent.
