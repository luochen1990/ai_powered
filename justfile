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
