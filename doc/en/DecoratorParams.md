# Decorator Parameters Design

Design decisions for parameterizing the `@ai_powered` decorator (`model`, `retry`, `agent`).
This document is the SSOT for these decisions; later batches (model registry, agent protocol)
must conform to it.

## Decisions (locked)

| # | Topic | Decision |
|---|-------|----------|
| D1 | `model` format | String only, provider-prefixed: `"{provider}/{model_name}"` (e.g. `"deepseek/deepseek-chat"`). A bare name resolves through the legacy env path; ambiguous bare names fail. Inline config objects are **not** accepted (may be revisited later, no current need). |
| D2 | `retry` semantics | `retry = N` means **N extra attempts** after the initial one (default 0 = old fail-fast behavior). Plain re-execution only — **no** error-feedback loop (feeding validation errors back to the LLM is out of scope). Retryable failures: network/API errors (`openai.APIError`), LLM gateway transient faults (`LlmGatewayError`, i.e. gateways relaying upstream failures like 503 as HTTP 200 + body.error), malformed JSON / schema validation failures (`msgspec.DecodeError`, which covers `ValidationError`), and adapter structure assertions (`AssertionError`). Non-retryable exceptions propagate immediately. When exhausted, the **last exception is re-raised as-is** (no wrapper exception). Caveat: the `AssertionError` leg relies on `assert` statements in adapters, so under `python -O` those failures degrade to non-retryable errors. |
| D3 | `agent` form | Class only (`type[FunctionAgent]`), not instances. Constructed once per decorated function at decoration time. Orthogonal to `retry`: with an agent, `retry` re-runs the **whole agent task** (not individual LLM calls inside it). |
| D4 | Sandbox | Deferred. Will be added incrementally later as an attribute of agents, not a decorator parameter. |
| D5 | Non-goals | Inline model config objects; error-feedback retry; per-call dynamic model switching; cross-call memory/state; general-purpose agent framework features. |

## Layering note for `retry` (D2)

The library-level retry loop wraps *simulation + validation* as one unit. Network-level retries
of the OpenAI SDK (`max_retries` on `LlmConnection`, default 2) are independent and may stack:
worst case `(1 + retry) * (1 + sdk_max_retries)` low-level requests. This is documented
behavior, not handled in code.

## Implementation batches

- **B1 (done): decorator params + retry + offline test injection point.**
  - Dual-form decorator: bare `@ai_powered` and `@ai_powered(retry=N)`, both on sync and async functions. Backwards compatible.
  - Retry loop implemented as module-level `_call_with_retry` / `_simulate_and_validate` (easy-sync recompiles function bodies for sync execution, so closures and inner functions are **not** visible in the rewritten body — dependencies must be passed explicitly).
  - `FakeLlmConnection` (scripted responses/exceptions, `sync_compatible` like the real one) enables deterministic offline tests; the fixture also pins decoration-time model config so tests are independent of host env vars.
- **B2 (done): model registry + `model=` parameter + config SSOT consolidation.**
  - `llm/registry.py`: provider-name index over the single `KnownPlatform` dataset (`llm/known_models.py`); `register_model()` for user-defined providers (env-indirect API keys, no provider-name shadowing).
  - `llm/resolver.py`: the only model-resolution entry (`resolve_model`), consumed by both `@ai_powered` and `ChatBot`; prefixed names take the registry path, bare/None take the unchanged env path. Prefixed resolution applies **only** to the `model=` param — env `OPENAI_MODEL_NAME` stays pass-through (deployments use `org/model`-shaped names that APIs accept as-is).
  - Known-platform entries gained `base_url` / `api_key_env` / `default_model` (open platforms like openrouter/localhost fall back for unlisted names; closed platforms fail fast). `complete_model_config` keeps historical behavior (default fallback = `default_model or known_model_list[0]`, all built-in `default_model` = their `[0]`).
  - Decision: **no cross-decoration connection caching yet** — openai clients are cheap pooled objects, while caching would complicate test injection (monkeypatched `resolve_model`); revisit when a real use case demands it.
- **B3 (planned): agent protocol + reference implementation.** `FunctionSpec` value object; `FunctionAgent` ABC with `run(arguments_json) -> str` returning `{"result": ...}`-shaped JSON; exit validation applies to agents unchanged (agents cannot bypass the type contract). Sandbox out of scope (D4).

## Non-goals (guard against scope creep)

See D5 above. Anything beyond it needs a new decision record in this file.
