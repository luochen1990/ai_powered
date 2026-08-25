'''
职责: GenericAgent 的离线单元测试 —— 脚本化 FakeSandbox + conftest 的 ScriptedCompletionConnection 驱动
(全程无 bwrap 依赖: _backend_ready / _create_sandbox 均为测试子类的注入点).

覆盖:
  - 正常链路: bash 工具执行 → 回执回喂 (tool role) → return_result 提交
  - 工具面: read_file/write_file 精确往返 / google_search 有 key 才注册
  - 错误回喂自修复: 执行失败 / 超时 / 路径越界 → 错误文本回喂, 模型修复后重试 (不烧 retry 预算)
  - bash timeout: 模型可指定, clamp 到 bash_timeout_limit 硬上限
  - 未知工具 / custom tool call 变体 / max_rounds 耗尽 → AgentRunError
  - 沙盒生命周期: run 内共享, run 结束清理; 构造期 fail-fast (后端缺失 / bwrap 探测)
'''
import os
from typing import Any, cast

import pytest
from easy_sync import sync_compatible

from conftest import FAKE_TOOLS_MODEL, ScriptedCompletionConnection, make_content_completion, make_spec, make_tool_call_completion

from ai_powered import GenericAgent
from ai_powered.agent import AgentRunError
from ai_powered.llm.connection import LlmConnection
from ai_powered.sandbox import ExecutionResult, Sandbox


class FakeSandbox(Sandbox):
    ''' 脚本化沙盒: 不起进程, bash 记录命令并按 script 出回执; 文件桥用真实实现 (内存 dir) '''

    def __init__(self, script: list[ExecutionResult | Exception] = []):
        self.script = list(script)
        self.seen_commands: list[str] = []
        self.seen_timeouts: list[float] = []
        self.cleaned = False
        super().__init__(share_net = False)  # 建真实 scratch dir (文件桥可实测), bash 走脚本

    @sync_compatible
    async def run_bash(self, script: str, timeout: float) -> ExecutionResult:
        self.seen_commands.append(script)
        self.seen_timeouts.append(timeout)
        step = self.script[min(len(self.seen_commands), len(self.script)) - 1] if self.script else ExecutionResult("", "", 0, False)
        if isinstance(step, Exception):
            raise step
        return step

    def cleanup(self) -> None:
        self.cleaned = True
        super().cleanup()


def _agent_with(sandbox: FakeSandbox, script, **classvars: Any) -> tuple[GenericAgent, ScriptedCompletionConnection]:
    ''' 构造注入 FakeSandbox 的 GenericAgent 子类实例 (先换档后构造: 构造期 fail-fast 被注入点绕开) '''
    conn = ScriptedCompletionConnection(script)

    class _ScriptedAgent(GenericAgent):
        _backend_ready = staticmethod(lambda: True)  # 离线: 绕开 bwrap 探测
        _create_sandbox = lambda self: sandbox  #type: ignore[method-assign]  # noqa: E731
    for key, value in classvars.items():
        setattr(_ScriptedAgent, key, value)

    return _ScriptedAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn)), conn


def test_happy_path_bash_then_submit():
    sb = FakeSandbox([ExecutionResult("42\n", "", 0, False)])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "expr 6 * 7"}'),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{"x": 6}').wait() == '{"result": 42}'
    assert sb.seen_commands == ["expr 6 * 7"]
    # 第 2 轮请求 = (system, user, assistant 工具调用, tool 回喂): 执行结果已进对话
    assert [m["role"] for m in conn.seen_messages[1]] == ["system", "user", "assistant", "tool"]
    assert "42" in str(conn.seen_messages[1][-1].get("content"))


def test_execution_error_fed_back_for_self_repair():
    sb = FakeSandbox([
        ExecutionResult("", "bash: line 1: foo: command not found", 127, False),
        ExecutionResult("ok\n", "", 0, False),
    ])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "foo"}'),
        make_tool_call_completion("bash", '{"command": "bar"}'),
        make_tool_call_completion("return_result", '{"result": 1}'),
    ])

    assert agent.run('{}').wait() == '{"result": 1}'
    assert len(sb.seen_commands) == 2
    fed = str(conn.seen_messages[1][-1].get("content"))  # 第 2 轮请求末条 = 第 1 次执行回喂
    assert "127" in fed and "command not found" in fed


def test_timeout_fed_back():
    sb = FakeSandbox([ExecutionResult("", "", -9, True)])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "sleep 999", "timeout": 5}'),
        make_tool_call_completion("return_result", '{"result": "gave up"}'),
    ])

    assert agent.run('{}').wait() == '{"result": "gave up"}'
    assert "timed_out" in str(conn.seen_messages[1][-1].get("content"))


def test_bash_timeout_clamped_to_limit():
    ''' 模型 per-call timeout 只能往下调: 超过硬上限被 clamp (硬上限 = bash_timeout_limit) '''
    sb = FakeSandbox([ExecutionResult("", "", 0, False)])
    agent, _ = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "x", "timeout": 9999}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ], bash_timeout_limit = 30.0)

    agent.run('{}').wait()
    assert sb.seen_timeouts == [30.0]

    sb2 = FakeSandbox([ExecutionResult("", "", 0, False)])
    agent2, _ = _agent_with(sb2, [
        make_tool_call_completion("bash", '{"command": "y", "timeout": 3}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ], bash_timeout_limit = 30.0)

    agent2.run('{}').wait()
    assert sb2.seen_timeouts == [3.0]  # 更短的合法指定原样通过


def test_bash_run_exception_fed_back():
    ''' run_bash 抛异常 (如后端缺失) → 文本回喂, 不炸 run '''
    sb = FakeSandbox([RuntimeError("bwrap is not installed; bubblewrap is required to run the sandbox")])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "ls"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    assert agent.run('{}').wait() == '{"result": 0}'
    assert "bubblewrap" in str(conn.seen_messages[1][-1].get("content"))


def test_read_write_file_tools_roundtrip():
    ''' read_file/write_file 工具: host 侧文件桥精确往返 (经回喂内容断言 —— run 结束沙盒已清理) '''
    sb = FakeSandbox([])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("write_file", '{"path": "in/a.txt", "content": "line1\\nline2"}'),
        make_tool_call_completion("read_file", '{"path": "in/a.txt"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    agent.run('{}').wait()
    # 第 2 轮请求末条 = write 回执 ("written: in/a.txt"); 第 3 轮末条 = read 回喂 (文件内容)
    assert "written: in/a.txt" in str(conn.seen_messages[1][-1].get("content"))
    assert "line2" in str(conn.seen_messages[2][-1].get("content"))


def test_write_file_escape_fed_back():
    ''' 路径越界 → ValueError 文本回喂, 不炸 run '''
    sb = FakeSandbox([])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("write_file", '{"path": "../escape.txt", "content": "x"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    assert agent.run('{}').wait() == '{"result": 0}'
    assert "ValueError" in str(conn.seen_messages[1][-1].get("content"))


def test_workspace_files_classvar_preset():
    ''' 用法2: workspace_files / share_net ClassVar → 真实 _create_sandbox 工厂按 ClassVar 装配沙盒 '''
    class _RealFactoryAgent(GenericAgent):  # 不覆写 _create_sandbox —— 直测真实工厂
        _backend_ready = staticmethod(lambda: True)
        workspace_files = {"seed.csv": "a,b\n1,2\n"}

    conn = ScriptedCompletionConnection([])
    agent = _RealFactoryAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))
    real_sandbox = agent._create_sandbox()  # Sandbox 构造不依赖 bwrap, 可离线直测
    try:
        assert real_sandbox.share_net is True  # GenericAgent 默认 ClassVar
        assert real_sandbox.read_file("seed.csv") == b"a,b\n1,2\n"  # 预置输入已写入
    finally:
        real_sandbox.cleanup()

    class _OfflineAgent(_RealFactoryAgent):
        share_net = False

    offline = _OfflineAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))
    s2 = offline._create_sandbox()
    try:
        assert s2.share_net is False
    finally:
        s2.cleanup()


def test_sandbox_persists_within_run_and_cleaned_after():
    sb = FakeSandbox([])
    agent, _ = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "echo x > f"}'),
        make_tool_call_completion("bash", '{"command": "cat f"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    agent.run('{}').wait()
    assert len(sb.seen_commands) == 2 and sb.cleaned  # run 结束即清理
    assert not os.path.isdir(sb.root)


def test_sandbox_cleaned_even_on_run_error():
    sb = FakeSandbox([])
    agent, _ = _agent_with(sb, [make_content_completion("thinking ...")])  # 永不提交 → AgentRunError

    with pytest.raises(AgentRunError):
        agent.run('{}').wait()
    assert sb.cleaned


def test_google_search_registered_only_with_key(monkeypatch):
    ''' google_search 是便利件: 无 GOOGLE_API_KEY 静默不注册 (工具面收窄, run 不炸) '''
    monkeypatch.delenv("GOOGLE_API_KEY", raising = False)
    sb = FakeSandbox([])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "true"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    agent.run('{}').wait()
    assert conn.seen_tools[0] == ["bash", "read_file", "write_file", "return_result"]

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-registration")
    sb2 = FakeSandbox([])
    agent2, conn2 = _agent_with(sb2, [
        make_tool_call_completion("bash", '{"command": "true"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])
    agent2.run('{}').wait()
    assert "google_search" in conn2.seen_tools[0]


def test_google_search_tool_executes_when_registered(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-registration")
    sb = FakeSandbox([])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("google_search", '{"keywords": "test", "options": {}}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    # google_search.fn 会真发 HTTP —— 拦截之 (只验证分发链路, 不触网)
    import ai_powered.agent as agent_mod
    class _FakeSearch:
        def __getattr__(self, name):
            return lambda **kw: [{"title": "t", "link": "l", "snippet": "s"}]
        def schema(self):
            from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
            return cast(ChatCompletionToolParam, {"type": "function", "function": {"name": "google_search", "parameters": {"type": "object", "properties": {}, "required": []}}})
    monkeypatch.setattr(agent_mod, "google_search", _FakeSearch())

    agent.run('{}').wait()
    fed = str(conn.seen_messages[1][-1].get("content"))
    assert '"title": "t"' in fed  # 搜索结果 json 的精确内容 (分发链路把 fn 返回值文本化回喂)


def test_unexpected_tool_fed_back():
    ''' 未知工具名 → 环内文本回喂自修复 (与一切工具失败同为反馈, 不烧 retry 预算; max_rounds 兜底失控) '''
    sb = FakeSandbox([])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("do_something_else", '{}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    assert agent.run('{}').wait() == '{"result": 0}'
    assert sb.seen_commands == []  # 未知工具未被执行
    assert "unknown tool" in str(conn.seen_messages[1][-1].get("content"))


def test_custom_tool_call_variant_rejected():
    ''' custom tool call 等非 Function 变体: 拒绝 (避免 "有 tool_calls 无 tool 响应" 的非法对话序列) '''
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall, Custom

    custom_tc = ChatCompletionMessageCustomToolCall(id = "call-c", custom = Custom(name = "bash", input = "x"), type = "custom")
    resp = ChatCompletion(
        id = "fake", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "tool_calls", index = 0, message = ChatCompletionMessage(role = "assistant", content = None, tool_calls = [custom_tc]))],
    )
    sb = FakeSandbox([])
    agent, _ = _agent_with(sb, [resp])

    with pytest.raises(AgentRunError, match = "unexpected tool call"):
        agent.run('{}').wait()
    assert sb.seen_commands == []  # custom 变体未被误执行


def test_max_rounds_exhausted():
    sb = FakeSandbox([])
    agent, _ = _agent_with(sb, [make_content_completion("thinking ...")], max_rounds = 3)  # 耗尽重复末元素: 永不提交

    with pytest.raises(AgentRunError, match = "3 rounds"):
        agent.run('{}').wait()


def test_construction_fails_fast_without_backend(monkeypatch):
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: None)
    conn = ScriptedCompletionConnection([])

    with pytest.raises(ValueError, match = "bubblewrap"):
        GenericAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))


def test_construction_ok_with_backend(monkeypatch):
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/bwrap")
    conn = ScriptedCompletionConnection([])

    agent = GenericAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))
    assert agent.bash_timeout_limit == 30.0 and agent.share_net is True  # 用法1 默认值


@pytest.mark.asyncio
async def test_agent_async_run_path():
    ''' async 路径: run 返回 Waitable, await 直接走原生协程体 (FakeSandbox 注入, 离线) '''
    sb = FakeSandbox([ExecutionResult("42\n", "", 0, False)])
    agent, _ = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "expr 6 * 7"}'),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert await agent.run('{"x": 6}') == '{"result": 42}'
    assert sb.seen_commands == ["expr 6 * 7"]
