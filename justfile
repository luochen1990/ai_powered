# justfile - ai_powered 项目的开发便利命令
#
# 职责划分:
#   - just check       : 一键质量检查 (类型 + lint + 离线单测), 不依赖外部服务
#   - just test        : 功能测试 (test/examples/, 走 ai_powered 代码全链路, 需要可用 LLM provider)
#   - just test-model  : 模型选型验证 (scripts/model_baseline.py, 绕开项目代码, 需要本地 ollama)
#   - just test-all    : 先验证模型可靠, 再跑功能测试 (推荐换模型时使用)

# NOTE: pytest 统一加 `-o pythonpath=src` —— worktree/共享 venv 场景下防止 import 解析到主仓库旧代码
# (共享 venv 的 site-packages 可能 editable-install 指向主仓库 src, 项目外的 python -c / pyright 探针同样会被劫持,
#  需显式 PYTHONPATH=src / extraPaths); 该配置不落 pyproject.toml, 否则 CI 的 wheel 安装测试分支 (version: 前缀提交)
# 会被 checkout 源码劫持, 门禁静默失效
pytest_cmd := "poetry run pytest -o pythonpath=src"

# 默认列出可用命令
default:
    @just --list

# 质量检查: 类型检查 + lint + 离线单元测试 (不依赖外部服务, CI 对齐)
check:
    poetry run pyright -p pyproject.toml
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    {{pytest_cmd}} test/unit -q

# 功能测试: 验证 ai_powered 代码功能 (test/examples/)
# 需要先 source 一个可用的 provider envrc (如 kimi/openai/ollama)
# 默认使用 ollama 本地基准模型 (见 ollama.private.envrc)
test *args='':
    source ./ollama.private.envrc && {{pytest_cmd}} {{args}}

# 用指定 provider 跑功能测试
# 示例: just test-provider kimi
test-provider provider *args='':
    source ./{{provider}}.private.envrc && {{pytest_cmd}} {{args}}

# 模型选型验证: 验证候选模型能否作为测试基准 (scripts/model_baseline.py)
# 直接调 OpenAI 兼容 HTTP 端点, 绕开 ai_powered 代码, 是 test 的前置门禁
# 默认使用 ollama.private.envrc 配置的模型
test-model *args='':
    source ./ollama.private.envrc && poetry run python scripts/model_baseline.py {{args}}

# 一键验证: 先确认模型可靠, 再跑功能测试 (换模型时推荐)
test-all:
    @echo "=== Step 1: 模型选型验证 ==="
    just test-model
    @echo "=== Step 2: 功能测试 ==="
    just test

# org ci-merge 契约 recipe (阻塞级门禁入口, 见 lc-studio/forgejo-actions README)。
# CI 判据用 CI env (runner 注入), 不用 VIRTUAL_ENV — 后者是通用 venv 激活变量,
# 本地开发者激活自己的 venv 跑 just ci-merge 会被误判进 CI 分支 (且戳块 rm -rf 会
# 销毁其 venv)。本地 (无 CI env) 退化为 poetry run 直调, 不做 venv 复用。
ci-merge:
    #!/usr/bin/env bash
    set -eu
    # pytest 参数单点定义: use_google_search 直连 googleapis.com, runner VM 不出境
    # (SYN-SENT 挂起, 曾致 CI 30min 超时) 故 deselect; reruns 3x20s 抗 LLM 抖动
    pytest_args=(--reruns 3 --reruns-delay 20 --deselect test/examples/chat_bot/use_google_search.py)
    if [ -n "${CI:-}" ]; then
        # VIRTUAL_ENV 由 workflow env 指向持久卷; 丢失/GC 由戳块自愈重建
        STAMP="$VIRTUAL_ENV.python-stamp"
        WANT="$(python3.13 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [ ! -x "$VIRTUAL_ENV/bin/python" ] || [ "$(cat "$STAMP" 2>/dev/null || true)" != "$WANT" ]; then
            rm -rf "$VIRTUAL_ENV"
            python3.13 -m venv "$VIRTUAL_ENV"
            printf '%s\n' "$WANT" > "$STAMP"
            echo "venv: created fresh ($WANT)"
        else
            echo "venv: reuse ($WANT)"
        fi
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        poetry sync --with test --without dev
        # -p pyproject.toml: 防 worktree 相对路径外 pyrightconfig 劫持 (同 check recipe)
        pyright -p pyproject.toml
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        pytest "${pytest_args[@]}"
    else
        just check
        poetry run pytest "${pytest_args[@]}"
    fi
