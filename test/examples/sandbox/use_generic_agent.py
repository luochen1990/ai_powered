'''
职责: @ai_powered(agent=GenericAgent) 的真实 LLM 端到端示例 —— 沙盒内 fs 读写 + 网络访问 + rg 全链路.

需要本机安装 bubblewrap (Linux) 与网络可达. 装饰在模块 import (= pytest collection) 期构造
agent 并探测后端, 故守卫是模块级 if 而非 skipif (后者求值太晚, 无 bwrap 环境收集期即错).
'''
import shutil

from ai_powered import GenericAgent, ai_powered

if shutil.which("bwrap") is not None:

    @ai_powered(agent = GenericAgent, retry = 1)
    def fetch_and_summarize(task: str) -> str:
        ''' Complete the task autonomously in the sandboxed workspace (files, bash, network). '''
        ...

    def test_fetch_store_and_grep():
        answer = fetch_and_summarize(
            "Download https://example.com into page.html in your workspace, "
            "then use rg to extract the page <title> ... </title> line, and return the title text."
        )
        print(f"{answer =}")
        assert "Example Domain" in answer
