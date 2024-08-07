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
