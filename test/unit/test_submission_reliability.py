'''
职责: 交卷可靠性机制 (issue #28) 的离线单元测试 —— 失速注入 × 指数收缩水位提醒 × 提交环内校验回喂.

覆盖 issue #28 的四类验证场景 + 决策矩阵关键交点:
  - 事件轴 (失速检测): 首轮纯文本放任 / 连续 ≥2 轮注入交卷提醒 / 计数器语义 (工具轮清零, 空响应计入) /
    ReasoningAgent 关闭失速注入 (连续纯文本推理是其设计意图) / 空响应立即追问
  - 水位轴 (预算校准): 恰在 r = N/2, N/4, ... 触发, 其余轮零注入 (半程零打扰, 尺度不变) /
    轴独立运行 (工具轮也注入额度提醒) / 两轴内容叠加为单条消息 / r=1 强制 tool_choice 终点
  - 交卷环 (校验回喂): 非法提交逐字段错误回喂后下一轮修复 (不烧 retry) /
    并行批每个 tool_call 各有响应 (OpenAI 协议) / 批内合法提交免执行同批其余调用
  - 收编通道: 纯文本整体恰为合法 Result[R] JSON 直接收下 / 夹带 JSON 片段的推理文本不误收

驱动方式: GenericAgent 用 test_generic_agent 的 FakeSandbox + _agent_with (离线);
ReasoningAgent / _SubmissionWatchdog 无沙盒依赖, 直接构造.
'''
from typing import cast

import pytest

from conftest import FAKE_TOOLS_MODEL, ScriptedCompletionConnection, make_content_completion, make_spec, make_tool_call_completion, make_tool_calls_completion
from test_generic_agent import FakeSandbox, _agent_with

from ai_powered import ReasoningAgent
from ai_powered.agent import AgentRunError, _SubmissionWatchdog
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec
from ai_powered.sandbox import ExecutionResult


# ---------- 水位轴: watchdog 状态机 (纯单元) ----------

@pytest.mark.parametrize("total,expected", [(2, [1]), (3, [1]), (8, [4, 2, 1]), (16, [8, 4, 2, 1]), (50, [25, 12, 6, 3, 1])])
def test_watermark_trigger_rounds(total: int, expected: list[int]):
    ''' 水位轴性质: 触发轮恰为逐次减半序列 (r ≤ 上次提醒剩余/2, 首次在 N/2), 其余轮零注入; 尺度不变零调参 '''
    watchdog = _SubmissionWatchdog(total, stall_threshold = None, stall_reminder = "x")
    triggered = [r for r in range(total - 1, 0, -1) if watchdog.round_end_injection(r, []) is not None]
    assert triggered == expected


def test_watchdog_stall_counter_semantics():
    ''' 事件轴计数器: 失速轮累计, 工具轮清零; 空响应无条件追问且计入失速; 无下一轮 (r_next=0) 时事件提醒也不注入 (无人消费) '''
    watchdog = _SubmissionWatchdog(8, stall_threshold = 2, stall_reminder = "REMIND")
    assert watchdog.note_stall_round() is None  # 连续第 1 轮: 放任 (可能是规划)
    assert watchdog.note_stall_round() == "REMIND"  # 连续第 ≥2 轮: 失速
    watchdog.note_tool_round()  # 工具轮: 计数清零
    assert watchdog.note_stall_round() is None

    threshold_free = _SubmissionWatchdog(8, stall_threshold = None, stall_reminder = "NUDGE")
    assert threshold_free.note_empty_round() == "NUDGE"  # 空响应: 阈值天然为 1, 不依赖 stall_threshold
    assert threshold_free.stall_count == 1  # 且计入失速轮
    assert _SubmissionWatchdog(8, stall_threshold = 2, stall_reminder = "x").round_end_injection(0, ["EVENT"]) is None


def test_round_end_injection_merges_axes():
    ''' 轴叠加: 事件提醒 + 水位提醒合并为单条消息; r=1 为最后通牒措辞 (数据说话) '''
    watchdog = _SubmissionWatchdog(4, stall_threshold = None, stall_reminder = "x")
    assert watchdog.round_end_injection(2, ["EVENT"]) == "EVENT\n2 of 4 rounds remaining."
    assert watchdog.round_end_injection(3, []) is None  # 已提醒至 2: 2*3 > 2 不再触发

    ultimatum = _SubmissionWatchdog(4, stall_threshold = None, stall_reminder = "x").round_end_injection(1, [])
    assert ultimatum is not None and "last round" in ultimatum and "return_result" in ultimatum


# ---------- 水位轴: agent run 集成 ----------

def test_watermark_exponential_sequence_and_forced_endpoint():
    ''' 尾轮预算: N=16 全程纯文本 → 恰在第 9/13/15/16 次请求看到水位注入 (r=8,4,2,1), 其余轮末条零注入;
    tool_choice 仅末轮强制 (ReasoningAgent: 失速注入关闭, 水位轴兜底) '''
    conn = ScriptedCompletionConnection([make_content_completion("thinking ...")])  # 耗尽重复末元素

    class _N16(ReasoningAgent):
        max_rounds = 16

    agent = _N16(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))
    with pytest.raises(AgentRunError):
        agent.run('{}').wait()

    assert conn.call_count == 16
    expected: dict[int, str] = {9: "8 of 16 rounds remaining.", 13: "4 of 16 rounds remaining.", 15: "2 of 16 rounds remaining."}
    for idx in range(1, 17):
        last = conn.seen_messages[idx - 1][-1]
        if idx in expected:
            assert last["role"] == "user" and str(last.get("content")) == expected[idx], f"request {idx}"
        elif idx == 16:
            assert last["role"] == "user" and "last round" in str(last.get("content"))
        elif idx == 1:
            assert [m["role"] for m in conn.seen_messages[0]] == ["system", "user"]  # 初始对话原样 (零注入)
        else:
            assert last["role"] == "assistant", f"request {idx} 应零注入 (半程零打扰 + 指数间隔)"
    assert [tc is not None for tc in conn.seen_tool_choice] == [False] * 15 + [True]


def test_generic_agent_forced_choice_at_last_round():
    ''' GenericAgent 同款强制终点: 仅 r=1 的请求携带 tool_choice = return_result '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion("thinking"),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ], max_rounds = 2)

    assert agent.run('{}').wait() == '{"result": 0}'
    assert conn.seen_tool_choice == [None, {"type": "function", "function": {"name": "return_result"}}]


def test_watermark_applies_to_tool_rounds():
    ''' 决策矩阵 "带 tool_calls × 触发水位": 轴独立运行 —— 工具轮无事件提醒, 但额度提醒照注入 (tool 响应之后) '''
    sb = FakeSandbox([ExecutionResult("1", "", 0, False), ExecutionResult("2", "", 0, False)])
    agent, conn = _agent_with(sb, [
        make_tool_call_completion("bash", '{"command": "a"}'),
        make_tool_call_completion("bash", '{"command": "b"}'),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ], max_rounds = 4)

    agent.run('{}').wait()
    msgs3 = conn.seen_messages[2]
    assert msgs3[-2]["role"] == "tool" and msgs3[-1]["role"] == "user"  # tool 响应后紧跟 user 水位消息
    assert str(msgs3[-1].get("content")) == "2 of 4 rounds remaining."


def test_watermark_stacks_with_empty_nudge():
    ''' 决策矩阵 "空响应 × 触发水位": 追问 + 额度叠加为单条 user 消息 '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion(None),
        make_content_completion(None),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ], max_rounds = 4)

    assert agent.run('{}').wait() == '{"result": 0}'
    content = str(conn.seen_messages[2][-1].get("content"))
    assert "sandbox tools" in content and "return_result" in content  # 事件轴: 空响应追问 (阈值天然为 1)
    assert "2 of 4 rounds remaining." in content  # 水位轴: 叠加


# ---------- 事件轴: 失速注入 ----------

def test_consecutive_text_stall_injects_reminder():
    ''' 失速作答: 首轮纯文本放任 (可能是规划), 连续第 2 轮注入交卷提醒 (模型不是没完成, 是结束意图没转化为交卷动作) '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion("Let me plan: I will inspect the files first."),
        make_content_completion("The answer is obviously 42."),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert conn.seen_messages[1][-1]["role"] == "assistant"  # 第 2 次请求零注入 (首轮放任)
    last = conn.seen_messages[2][-1]
    assert last["role"] == "user" and "return_result" in str(last.get("content"))  # 第 3 次请求末条 = 交卷提醒


def test_stall_counter_resets_on_tool_round():
    ''' 计数器语义: 出现 tool_call → 清零; 重置后首个纯文本轮重新放任 '''
    sb = FakeSandbox([ExecutionResult("done", "", 0, False)])
    agent, conn = _agent_with(sb, [
        make_content_completion("plan ..."),  # c=1 放任
        make_tool_call_completion("bash", '{"command": "ls"}'),  # c=0
        make_content_completion("hmm ..."),  # c=1 放任 (第 4 次请求零注入)
        make_content_completion("still thinking ..."),  # c=2 → 提醒
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    agent.run('{}').wait()
    assert conn.seen_messages[3][-1]["role"] == "assistant"
    assert conn.seen_messages[4][-1]["role"] == "user"


def test_empty_round_counts_toward_stall():
    ''' 计数器语义: 空响应计入失速轮 —— empty(c=1, 追问) → text(c=2, 交卷提醒) '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion(None),
        make_content_completion("thinking some more"),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    agent.run('{}').wait()
    assert [m["role"] for m in conn.seen_messages[2]] == ["system", "user", "assistant", "user", "assistant", "user"]  # 追问 + 失速提醒两不缺


def test_reasoning_agent_stall_injection_disabled():
    ''' ReasoningAgent 差异化: 连续纯文本是设计意图 → 失速注入关闭 (无交卷提醒), 水位轴照常兜底 '''
    conn = ScriptedCompletionConnection([make_content_completion("reasoning ...")])  # 耗尽重复

    agent = ReasoningAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))
    with pytest.raises(AgentRunError):
        agent.run('{}').wait()

    for idx in (2, 3, 4, 6):  # N=8: 水位注入仅在 5/7/8; 其余请求末条为 assistant (纯文本推理无打扰)
        assert conn.seen_messages[idx - 1][-1]["role"] == "assistant", f"request {idx}"
    assert str(conn.seen_messages[4][-1].get("content")) == "4 of 8 rounds remaining."


def test_generic_agent_empty_round_nudge_preserved():
    ''' 既有行为锚: 空响应且无工具调用 → 立即追问 (阈值天然为 1, 保留 role 交替) '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion(None),
        make_tool_call_completion("return_result", '{"result": 0}'),
    ])

    assert agent.run('{}').wait() == '{"result": 0}'
    assert [m["role"] for m in conn.seen_messages[1]] == ["system", "user", "assistant", "user"]


# ---------- 交卷环: 提交环内校验回喂 ----------

def test_invalid_submission_fed_back_and_fixed_in_loop():
    ''' 非法提交: ValidationError 文本作为该 tool_call 的响应回喂, 下一轮修复交卷 —— 修复成本从烧一次 retry 降为 1 轮环内 '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_tool_call_completion("return_result", '{"result": "not-an-int"}'),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    msgs2 = conn.seen_messages[1]
    assert [m["role"] for m in msgs2] == ["system", "user", "assistant", "tool"]
    fed = msgs2[-1]
    assert fed.get("tool_call_id") == "call-1"  # 对位回喂
    content = str(fed.get("content"))
    assert "Invalid submission" in content and "result" in content  # 逐字段错误 (自带 JSON 路径 $.result)


def test_reasoning_agent_invalid_submission_fed_back():
    ''' ReasoningAgent 同款交卷环: 非法提交不烧 run, 环内修复 '''
    conn = ScriptedCompletionConnection([
        make_tool_call_completion("return_result", '{"result": "not-an-int"}'),
        make_tool_call_completion("return_result", '{"result": 7}'),
    ])
    agent = ReasoningAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))

    assert agent.run('{}').wait() == '{"result": 7}'
    assert [m["role"] for m in conn.seen_messages[1]] == ["system", "user", "assistant", "tool"]
    assert "Invalid submission" in str(conn.seen_messages[1][-1].get("content"))


def test_reasoning_agent_mixed_batch_with_unknown_tool_rejected():
    ''' 既有语义锚: 混合批 [未知工具, 非法 return_result] 无合法提交 → 硬失败 (环内修复仅覆盖纯提交批;
    ReasoningAgent 无执行引擎, 未知工具维持 AgentRunError) '''
    conn = ScriptedCompletionConnection([
        make_tool_calls_completion([("call-0", "do_something_else", "{}"), ("call-1", "return_result", '{"result": "bad"}')]),
    ])
    agent = ReasoningAgent(make_spec(), FAKE_TOOLS_MODEL, cast(LlmConnection, conn))

    with pytest.raises(AgentRunError, match = "do_something_else"):
        agent.run('{}').wait()


def test_parallel_batch_every_tool_call_answered():
    ''' 并行协议: [bash, 非法 return_result] 同批 → 每个调用各有 tool 响应且按原始调用顺序 (OpenAI 协议要求) '''
    sb = FakeSandbox([ExecutionResult("42\n", "", 0, False)])
    agent, conn = _agent_with(sb, [
        make_tool_calls_completion([("call-0", "bash", '{"command": "expr 6 * 7"}'), ("call-1", "return_result", '{"result": "bad"}')]),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert sb.seen_commands == ["expr 6 * 7"]  # 同批普通工具照常执行
    fed_tools = [m for m in conn.seen_messages[1] if m["role"] == "tool"]
    assert [m.get("tool_call_id") for m in fed_tools] == ["call-0", "call-1"]
    assert "42" in str(fed_tools[0].get("content"))
    assert "Invalid submission" in str(fed_tools[1].get("content"))


def test_valid_submission_skips_sibling_executions():
    ''' 批内已有合法提交: 同批其余调用无执行必要 (结果无人消费, run 即将结束) '''
    sb = FakeSandbox([])
    agent, _ = _agent_with(sb, [
        make_tool_calls_completion([("call-0", "bash", '{"command": "side-effect"}'), ("call-1", "return_result", '{"result": 42}')]),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert sb.seen_commands == []


# ---------- 收编通道 ----------

def test_whole_json_text_incorporated():
    ''' 收编: 纯文本整体恰为合法 Result[R] JSON → 直接收下 (模型明确给了答案只是走错通道) '''
    agent, conn = _agent_with(FakeSandbox([]), [make_content_completion('{"result": 42}')])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert conn.call_count == 1


def test_json_with_surrounding_whitespace_incorporated():
    ''' 收编边界: 严格整体 decode 容忍首尾空白 (msgspec JSON 语义) '''
    agent, _ = _agent_with(FakeSandbox([]), [make_content_completion(' {"result": 42}\n')])

    assert agent.run('{}').wait() == ' {"result": 42}\n'  # 原文返回


def test_text_with_embedded_json_not_incorporated():
    ''' 禁止子串提取: 推理文本夹带 JSON 片段不误收 (否则模型 "讨论要生成的 JSON" 会被错误提前终止) '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion('The JSON should look like {"result": 42} per the schema.'),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert conn.call_count == 2  # 未被误收, 继续循环


def test_fenced_json_not_incorporated():
    ''' 禁止子串提取: markdown 围栏包裹的 JSON 同样不误收 '''
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion('```json\n{"result": 42}\n```'),
        make_tool_call_completion("return_result", '{"result": 42}'),
    ])

    assert agent.run('{}').wait() == '{"result": 42}'
    assert conn.call_count == 2


# ---------- 降级路径 ----------

def test_no_annotation_degrades_to_exit_validation():
    ''' spec 未带返回值注解 (直构场景): 环内校验/收编降级直通 —— 结构化校验交由装饰器出口兜底 '''
    bare_spec = FunctionSpec("fn", "def fn(x: int) -> int", "doc", {"x": {"type": "integer"}}, {"type": "integer"})
    agent, conn = _agent_with(FakeSandbox([]), [
        make_content_completion('{"result": 1}'),  # 收编关闭: 不识别
        make_tool_call_completion("return_result", '{"result": "type-mismatch-passes-through"}'),  # 校验关闭: 直通
    ], spec = bare_spec)

    assert agent.run('{}').wait() == '{"result": "type-mismatch-passes-through"}'
    assert conn.call_count == 2
