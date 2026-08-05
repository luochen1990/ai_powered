# justfile - ai_powered 项目的开发便利命令
#
# 职责划分:
#   - just check       : 一键质量检查 (类型 + lint), 不依赖外部服务
#   - just test        : 功能测试 (test/examples/, 走 ai_powered 代码全链路, 需要可用 LLM provider)
#   - just test-model  : 模型选型验证 (scripts/model_baseline.py, 绕开项目代码, 需要本地 ollama)
#   - just test-all    : 先验证模型可靠, 再跑功能测试 (推荐换模型时使用)

# 默认列出可用命令
default:
    @just --list

# 质量检查: 类型检查 + lint (不依赖外部服务, CI 对齐)
check:
    poetry run pyright
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 功能测试: 验证 ai_powered 代码功能 (test/examples/)
# 需要先 source 一个可用的 provider envrc (如 kimi/openai/ollama)
# 默认使用 ollama 本地基准模型 (见 ollama.private.envrc)
test *args='':
    source ./ollama.private.envrc && poetry run pytest {{args}}

# 用指定 provider 跑功能测试
# 示例: just test-provider kimi
test-provider provider *args='':
    source ./{{provider}}.private.envrc && poetry run pytest {{args}}

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
