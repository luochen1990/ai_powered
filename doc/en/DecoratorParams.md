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
| D4 | Sandbox | Shipped in B4 as an attribute of agents: `GenericAgent` owns a private bwrap-backed sandbox (Linux-only, pinned decision). No executor protocol, no decorator parameter — configuration via subclass ClassVars (`bash_timeout_limit` / `share_net` / `workspace_files`). |
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
- **B3 (done): agent protocol + reference implementation.**
  - `FunctionSpec` value object (`llm/definitions.py`): the 5 scattered simulator fields extracted once at decoration time; shared SSOT between single-request simulators and agents.
  - `FunctionAgent` ABC (`agent.py`): constructed once per decorated function with `(spec, model_config, connection)`; subclass contract is a `@sync_compatible async def run(arguments_json) -> str` returning `{"result": ...}`-shaped JSON. Protocol enforcement at decoration time: non-class agents, bare `async def run` (missing `@sync_compatible` — breaks the sync-translated `.wait()` chain), and `required_features` not covered by the model's capabilities all fail fast with `ValueError`. Exit validation applies to agents unchanged (agents cannot bypass the type contract).
  - `_AgentEngine` adapter (`decorators.py`): adapts `agent.run` to the `query_model` shape so agents reuse the same retry/validate pipeline; `retry` re-runs the **whole agent run** (orthogonality pinned by tests).
  - `AgentRunError` (retryable): multi-round budget exhaustion / unexpected tool calls are sampling-noise failures of the same class as the single-request `AssertionError`, so they are retryable; dedicated type instead of bare `RuntimeError` to avoid retrying unrelated programming errors (precedent: `LlmGatewayError`).
  - `ReasoningAgent` reference implementation: multi-turn reasoning loop — the model may emit plain-text reasoning rounds, then submits via the shared `return_result` tool; empty rounds get a nudge (with assistant placeholder to keep role alternation); `max_rounds` guards runaway loops; parallel tool calls pick `return_result` and ignore others. `return_result_tool_param` extracted to `tool_call.py` as the SSOT of the submission contract (shared with `ToolsFunctionSimulator`); `base_system_prompt` extracted to `constants.py` as the SSOT of the common prompt base (shared by adapters and agents). The decorators-side debug `sys_prompt` block was removed (it was a stale copy of adapter logic that mis-reported prompts for structured-output models and was meaningless on the agent path).
  - Sandbox remains out of scope (D4) — planned as an attribute of agents, incrementally.
- **B4 (done): sandbox — `GenericAgent` (bwrap-backed, sandbox as an agent attribute).**
  - `GenericAgent` (`agent.py`): general-purpose agent — the model works autonomously in a sandbox via `bash` (primary; `rg` instead of grep, per decision), `read_file`/`write_file` (exact round-trips over the host-side file bridge), optional `google_search` (host-side, registered only when `GOOGLE_API_KEY` is set), and `return_result`. All tool failures (exit≠0, timeouts, path escapes, backend errors) are fed back as text for in-loop self-repair instead of burning retry budget. Two usage forms: use the class directly (empty sandbox, defaults) or subclass and override ClassVars (`bash_timeout_limit` = default & hard cap for the model's per-call timeout, `share_net`, `workspace_files` preset inputs, `max_rounds`).
  - `sandbox.py` (private backend, no public exports beyond the agent): bubblewrap mount-namespace sandbox modeled after the author's `sandbox-exec` tool — layered security: `--unshare-all` [+ `--share-net`] → `--cap-drop ALL` → `--clearenv` + rebuilt PATH/HOME/TMPDIR (API keys unreachable by construction) → host fs invisible by default (ro-bind system paths; NixOS gets the whole `/nix/store` ro with symlink-resolved PATH) → child-side preexec rlimits (CPU/AS/FSIZE, soft=hard; NPROC deliberately excluded — it counts per host-UID and would break bwrap itself) → host-side timeout with process-group kill + `--die-with-parent`. Network mode additionally ro-binds CA/DNS (`/etc/ssl/certs`, `resolv.conf`, `hosts`) and points `SSL_CERT_FILE`/`CURL_CA_BUNDLE` at the symlink-resolved bundle (NixOS).
  - Command list is **cognitive-surface auditing** (pinned): the SSOT table (`COMMAND_GROUPS`) only governs what the model is told (bash tool docstring overview + `ls-commands`, a shim rendering `commands.md`, mounted read-only at `/commands` — deliberately NOT inside the workspace). Commands outside the list, if guessed, still cannot escape the bwrap boundary — matching sandbox-exec's "isolate namespaces/fs/net, not commands" philosophy. No constraint-surface auditing (per-package bind infrastructure, ~zero security gain).
  - Sandbox lifecycle = one agent run: created at run start (preset files written in), shared across tool calls within the run, cleaned up in `finally`.
  - Honest isolation level: mount-namespace tier — guards against mistakes and prompt-injection-induced destructive code, not a hard boundary against dedicated attackers (same tier as Flatpak). With `share_net=True` (default), data exfiltration and pip-style supply-chain risks are inherent to having network; documented, and turnable off via subclassing (`share_net = False`).
  - `utils/safe_eval.py` stays as-is for now (its known attribute-chain escape is real but out of this batch's scope; a dedicated fix PR is planned — search keyword: safe_eval).

## Non-goals (guard against scope creep)

See D5 above. Anything beyond it needs a new decision record in this file.
