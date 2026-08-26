# import 副作用守护 — 无凭据环境 import ai_powered 不得崩溃
#
# 背景: connection.py 类定义期用 wraps_method_arguments_type(openai.AsyncOpenAI()...)
# 提取签名, 裸构造在无 OPENAI_API_KEY 时抛错 → import 即崩, 下游 (llm-bench 测试/CLI) 全链受染。
# 修复: 签名载体带显式 dummy key (见 connection.py 注释)。本测试固化该契约。

import subprocess
import sys


def test_import_without_credentials():
    """空环境 (无 OPENAI_API_KEY/OPENAI_BASE_URL) import ai_powered 必须成功"""
    code = "import ai_powered; import ai_powered.llm.connection"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={},  # 完全清空环境变量: 任何 import 副作用都无处可藏
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
