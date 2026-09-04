'''
职责: FunctionAgent 协议 (函数执行引擎) + 参考实现 ReasoningAgent + 交卷可靠性机制 (issue #28).

协议定位 (doc/en/DecoratorParams.md D3):
  - @ai_powered(agent=SomeAgent) 用指定 agent 类替换默认的 "单次 LLM 请求" 模拟器;
    多轮推理/工具使用/代码执行等都只是引擎的实现细节, 出口校验 (Result[R] 解码) 对 agent 一视同仁,
    agent 不能绕过被装饰函数的类型契约
  - agent 与 retry 正交: retry 重试的是整个 agent run (而非 agent 内部单次 LLM 请求)
  - 沙盒 (执行 LLM 生成代码) 已按 D4 落地 (B4): GenericAgent 内化为 agent 属性 (私有 sandbox.py 后端)

交卷可靠性 (issue #28, 两个 agent 的 run 循环共用, 机制本体见 _SubmissionWatchdog 及其同族助手):
  - 事件轴 (响应特征): 失速检测 —— 连续纯文本/空响应达阈值注入交卷提醒, 工具轮静默;
    纯文本整体恰为合法 Result[R] JSON 时收编 (严格整体解析, 禁止子串提取)
  - 水位轴 (限额进度): 指数收缩提醒 (r ≤ 上次提醒剩余/2 触发, O(log N) 次注入) + r=1 强制 tool_choice 终点
  - 交卷环: return_result 提交即时按 Result[R] 校验, 失败逐字段错误回喂环内修复 (不烧 retry 预算)

实现约定 (easy-sync 约束, 见 decorators.py 的 NOTE):
  - 子类的 run 必须以 @sync_compatible 装饰的 async def 形式实现 —— 它会被 sync 转写体经 .wait() 调用
'''
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, cast

from easy_sync import sync_compatible
import asyncio
import json
import msgspec
import os
import shutil
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_named_tool_choice_param import ChatCompletionNamedToolChoiceParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ai_powered.constants import base_system_prompt
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import ModelConfig
from ai_powered.sandbox import COMMAND_GROUPS, ExecutionResult, Sandbox
from ai_powered.schema_deref import resolve_schema
from ai_powered.tool_call import Result, return_result_tool_param
from ai_powered.tools.google_search import google_search


class AgentRunError(RuntimeError):
    ''' agent run 的瞬时失败: 多轮预算耗尽 / 模型调用了未知工具等采样噪声类故障

    专用类型 (而非裸 RuntimeError) 是为了让 @ai_powered(retry=N) 能把它纳入可重试集合 ——
    与单次请求路径上 "响应不含期望结构" 的 AssertionError 同类; 不用裸 RuntimeError 是为
    避免误重试其它同名编程错误 (先例见 LlmGatewayError).
    '''


class FunctionAgent(ABC):
    ''' 函数执行引擎协议: 每个 @ai_powered(agent=...) 被装饰函数在装饰期构造一次, 每次调用执行一次 run

    子类契约:
    - __init__(spec, model_config, connection): 基类已保存这三者; 子类如需额外状态请 override 并调用 super().__init__
      (model 解析结果照常传入 —— agent 自主决定如何使用, 如用 connection 发多轮请求)
    - run(arguments_json) -> str: 以 @sync_compatible 装饰的 async def 实现, 返回 '{"result": ...}' 形态的 JSON 字符串
      (出口由装饰器统一按 Result[R] 校验, 无效输出会触发 retry / 抛出 DecodeError; 裸 async def 会被装饰期拒绝 ——
      sync 转写体经 .wait() 调用 run, 协程对象没有 .wait)
    - required_features: agent 对模型能力的硬依赖, 装饰期与 model_config.supported_features 求差, 不满足即 fail-fast
    '''

    spec: FunctionSpec
    model_config: ModelConfig
    connection: LlmConnection

    required_features: ClassVar[set[ModelFeature]] = set()

    #: 提交契约的解码类型 (Result[R], issue #28 环内校验/收编共用, 与出口校验同款);
    #: spec 未带返回值注解的直构场景为 None —— 环内校验/收编随之降级为直通, 校验交由出口兜底
    _result_type: type[Result[Any]] | None

    def __init__(self, spec: FunctionSpec, model_config: ModelConfig, connection: LlmConnection) -> None:
        self.spec = spec
        self.model_config = model_config
        self.connection = connection
        self._result_type = Result[spec.return_annotation] if spec.return_annotation is not None else None

    @sync_compatible
    @abstractmethod
    async def run(self, arguments_json: str) -> str:
        ''' 子类必须以 @sync_compatible async def run 覆盖 (见类 docstring 的 easy-sync 约定) '''


#: ReasoningAgent 的追问/交卷提醒措辞 (空响应追问与失速提醒共用 —— issue #28 "提醒措辞复用现有空响应追问")
_SUBMIT_REMINDER = "Please call the return_result tool to submit your result."

#: GenericAgent 的同款 (沙盒语境: 失速可能是 "只说不干", 提醒须同时给出干活与交卷两条出路)
_WORK_THEN_SUBMIT_REMINDER = "Please use the sandbox tools to work on the task, then call return_result to submit your result."

#: r=1 强制终点的 tool_choice (质变而非密度极限: 协议层硬约束独立于软提醒; 与单轮路径
#: ToolsFunctionSimulator._param_tool_choice_maker 对齐, 修复同仓机制不对称, issue #28)
_FORCED_RETURN_RESULT: ChatCompletionNamedToolChoiceParam = {"type": "function", "function": {"name": "return_result"}}


def _watermark_reminder(remaining: int, total_rounds: int) -> str:
    ''' 水位轴提醒文本: 数据说话 —— 倒计时数字本身即紧迫感信号, 每条携带增量信息, 模型无法习惯化 '''
    if remaining == 1:
        return f"1 of {total_rounds} rounds remaining. This is the last round: call the return_result tool now to submit your result."
    return f"{remaining} of {total_rounds} rounds remaining."


class _SubmissionWatchdog:
    ''' 交卷可靠性状态机 (issue #28): 事件轴 (失速计数) × 水位轴 (指数收缩提醒) 的每-run 可变状态

    轴分工: 失速检测归事件轴 (响应特征驱动 —— 工具轮静默, 连续纯文本/空响应计数), 预算校准归水位轴
    (限额进度驱动 —— r ≤ 上次提醒剩余 / 2 触发, 首次在 r = N/2, O(log N) 次注入, 零阈值参数, 尺度不变);
    两轴独立运行, 在 round_end_injection 处内容叠加为单条 user 消息, 唯一交点在 r=1 收敛为强制终点
    (tool_choice 由 run 循环施加, 本类只负责文本面).

    设计原则: 每条注入必须携带增量信息 (反 AgentScope 式逐字重复 reminder 的零信息通胀).
    '''

    def __init__(self, total_rounds: int, stall_threshold: int | None, stall_reminder: str) -> None:
        self.total_rounds = total_rounds
        #: None = 关闭失速注入 (ReasoningAgent: 连续纯文本推理是其设计意图, 交给水位轴兜底)
        self.stall_threshold = stall_threshold
        self.stall_reminder = stall_reminder
        self.stall_count = 0
        self.last_reminder = total_rounds  # 概念上 "上次提醒发生在 r=N" → 首次触发在 r=N/2 (半程零打扰)

    def note_tool_round(self) -> None:
        ''' 工具轮: 循环自驱 (act→observe→reason), 事件轴静默; 失速计数清零 '''
        self.stall_count = 0

    def note_stall_round(self) -> str | None:
        ''' 失速轮 (纯文本) 计数; 达阈值返回交卷提醒文本, 否则 None '''
        self.stall_count += 1
        if self.stall_threshold is not None and self.stall_count >= self.stall_threshold:
            return self.stall_reminder
        return None

    def note_empty_round(self) -> str:
        ''' 空响应轮: 计入失速 + 无条件追问 —— 无任何规划语义, 阈值天然为 1 (不依赖 stall_threshold) '''
        self.note_stall_round()
        return self.stall_reminder

    def round_end_injection(self, next_remaining: int, event_reminders: list[str]) -> str | None:
        ''' 端轮注入合成: 事件轴提醒 + 水位轴提醒合并为单条 user 消息文本; None = 本轮零注入

        next_remaining < 1 (无下一轮) 时零注入 —— 消息只有被下一轮请求看到才有意义.
        '''
        if next_remaining < 1:
            return None
        parts = list(event_reminders)
        if 2 * next_remaining <= self.last_reminder:
            parts.append(_watermark_reminder(next_remaining, self.total_rounds))
            self.last_reminder = next_remaining
        return "\n".join(parts) if parts else None


def _validate_submission(arguments_json: str, result_type: type[Result[Any]] | None) -> str | None:
    ''' 交卷环校验 (issue #28): 与出口 _simulate_and_validate 同款 Result[R] 解码

    通过 → None; 失败 → 逐字段错误文本 (回喂素材 —— 环内 1-2 轮修复, 替代烧一次 retry:
    retry 单元是整个 agent run, 含重建沙箱与重放全部轮次). ValidationError 为 DecodeError
    子类, 一并捕获; 错误消息自带 JSON 路径 (如 `$.result`). result_type 为 None → 放行 (出口兜底).
    '''
    if result_type is None:
        return None
    try:
        msgspec.json.decode(arguments_json, type = result_type)
        return None
    except msgspec.DecodeError as e:
        return f"Invalid submission ({type(e).__name__}: {e}). Fix the arguments and call return_result again."


def _try_incorporate(content: str, result_type: type[Result[Any]] | None) -> str | None:
    ''' 收编通道 (issue #28): 纯文本整体恰为合法 Result[R] JSON → 原文返回 (模型明确给了答案只是走错通道)

    严格整体 decode (msgspec 容忍首尾空白), 禁止子串提取 —— 模型推理中讨论 "要生成的 JSON
    长什么样" 不得被误提取 (错误提前终止).
    '''
    if result_type is not None and _validate_submission(content, result_type) is None:
        return content
    return None


def _scan_submission(tool_calls: list[ChatCompletionMessageFunctionToolCall], result_type: type[Result[Any]] | None) -> tuple[str | None, dict[str, str]]:
    ''' 交卷扫描 (issue #28 提交环): 校验批内全部 return_result 调用 (纯计算, 不执行其他工具)

    返回 (首个合法提交原文 or None, 各无效提交的回喂文本 by tool_call_id);
    合法提交存在时同批其余调用无需执行/回喂 —— 结果无人消费, run 即将结束.
    '''
    submission: str | None = None
    errors: dict[str, str] = {}
    for tool_call in tool_calls:
        if tool_call.function.name == "return_result":
            error = _validate_submission(tool_call.function.arguments, result_type)
            if error is None:
                if submission is None:
                    submission = tool_call.function.arguments
            else:
                errors[tool_call.id] = error
    return submission, errors


class ReasoningAgent(FunctionAgent):
    ''' 参考实现: 多轮推理引擎 —— 模型在对话中自由思考, 通过 return_result 工具提交结果

    与默认单次模拟的差别: 模型可以先输出若干轮纯文本推理 (不急于作答), 再调用工具提交结构化结果;
    max_rounds 上限防失控. 适用于需要 step-by-step 推理的任务. (需要模型支持 function calling)

    交卷可靠性 (issue #28): 失速注入对连续纯文本推理关闭 (stall_threshold = None, 水位轴兜底);
    水位轴指数收缩提醒 + r=1 强制 tool_choice 终点; return_result 提交环内校验回喂;
    纯文本整体恰为合法 Result[R] JSON 时直接收编.
    '''

    max_rounds: int = 8
    required_features: ClassVar[set[ModelFeature]] = {ModelFeature.tools}
    #: 连续失速轮的交卷提醒阈值 (None = 关闭; GenericAgent 用 2, 本类设计意图即连续纯文本推理)
    stall_threshold: ClassVar[int | None] = None

    @sync_compatible
    async def run(self, arguments_json: str) -> str:
        system_prompt = base_system_prompt(self.spec) + "\nThink step by step as you need, then submit your final result by calling the return_result tool.\n"
        # return_schema 为 raw 形态, 提交工具的参数 schema 按通道能力门控展开/透传 (issue 18, 与 GenericAgent.run 同款)
        refs_allowed = ModelFeature.schema_refs in self.model_config.supported_features
        tools = [return_result_tool_param(resolve_schema(self.spec.return_schema, refs_allowed = refs_allowed))]

        conversation: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": arguments_json},
        ]
        watchdog = _SubmissionWatchdog(self.max_rounds, type(self).stall_threshold, _SUBMIT_REMINDER)

        for _round in range(self.max_rounds):
            remaining = self.max_rounds - _round
            request_kwargs: dict[str, Any] = {"model": self.model_config.model_name, "messages": conversation, "tools": tools}
            if remaining == 1:  # 强制终点 (issue #28); 不支持 tool_choice 的通道忽略之, 由最后通牒措辞兜底
                request_kwargs["tool_choice"] = _FORCED_RETURN_RESULT
            response = await self.connection.chat_completions(**request_kwargs)

            message = response.choices[0].message
            event_reminders: list[str] = []

            tool_calls = message.tool_calls
            if tool_calls is not None:
                fn_tool_calls = [t for t in tool_calls if isinstance(t, ChatCompletionMessageFunctionToolCall)]
                submission, submission_errors = _scan_submission(fn_tool_calls, self._result_type)
                if submission is not None:  # 首个合法提交胜出, 同批其他调用按既有惯例忽略
                    return submission
                if fn_tool_calls and len(fn_tool_calls) == len(tool_calls) and all(t.function.name == "return_result" for t in fn_tool_calls):
                    # 纯提交批且均无效: 校验错误逐 call 回喂, 环内修复 (issue #28; 无执行引擎, 不涉其他工具);
                    # 空批 / 含未知工具的批不属此列 → 落入下方既有硬失败 (与旧语义一致)
                    watchdog.note_tool_round()
                    conversation.append(cast(ChatCompletionMessageParam, message.to_dict()))  # assistant 工具调用轮入栈
                    conversation.extend({"role": "tool", "tool_call_id": t.id, "content": submission_errors[t.id]} for t in fn_tool_calls)
                else:
                    called = [t.function.name for t in fn_tool_calls]
                    raise AgentRunError(f"ReasoningAgent got unexpected tool call(s): {called or tool_calls}")

            elif not message.content:  # 空响应且无工具调用: 无任何规划语义 → 立即追问 (阈值天然为 1); 计入失速轮
                conversation.append({"role": "assistant", "content": ""})  # 保留 assistant 占位以维持 role 交替惯例
                event_reminders.append(watchdog.note_empty_round())

            else:
                incorporated = _try_incorporate(message.content, self._result_type)
                if incorporated is not None:
                    return incorporated
                conversation.append(message.to_dict())  #type: ignore[arg-type]  # assistant 中间推理保留在对话中 (与 ChatBot 同款窄化惯例)
                reminder = watchdog.note_stall_round()
                if reminder is not None:
                    event_reminders.append(reminder)

            injection = watchdog.round_end_injection(remaining - 1, event_reminders)
            if injection is not None:
                conversation.append({"role": "user", "content": injection})

        raise AgentRunError(f"ReasoningAgent did not submit a result within {self.max_rounds} rounds")



def _tool_param(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ChatCompletionToolParam:
    ''' 工具 schema 构造小工厂 (GenericAgent 的 bash/read_file/write_file 用) '''
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}}


#: 命令概览由清单 SSOT 程序化展平 (认知面审核: 改 COMMAND_GROUPS 一处, docstring 与 ls-commands 同步)
_BASH_COMMANDS_OVERVIEW = " ".join(c for _, cmds in COMMAND_GROUPS for c in cmds)

#: 回喂内容的统一截断上限 (单点: 所有工具回执在入对话处统一尾部截断, 防长输出撑爆上下文)
_FEEDBACK_CHAR_LIMIT = 4000


def _run_google_search_sync(**kwargs: Any) -> str:
    ''' google_search 的阻塞实现 (host 侧执行, API key 经 env 引用 —— 密钥不进沙盒) '''
    return json.dumps(google_search.fn(**kwargs), ensure_ascii = False, default = str)


@sync_compatible(sync_fn = _run_google_search_sync)
async def _run_google_search(**kwargs: Any) -> str:
    ''' async 路径经 to_thread 卸载阻塞的 requests 调用 (双实现形态, 对齐 Sandbox.run_bash) '''
    return await asyncio.to_thread(_run_google_search_sync, **kwargs)


def _generic_agent_tools(timeout_limit: float) -> list[ChatCompletionToolParam]:
    ''' GenericAgent 的工具面: bash (主力) + 文件精确往返 + return_result; google_search 由调用侧按 API key 可用性附加 '''
    return [
        _tool_param(
            "bash", "Run a bash command in an isolated sandbox. The working directory is a writable isolated filesystem (HOME=/workspace); "
            "files persist across calls within this task. Known-good commands: " + _BASH_COMMANDS_OVERVIEW + ". "
            "Run `ls-commands` (or `ls-commands | rg <pattern>`) to list/search available commands with details. "
            "Returns exit_code with truncated stdout/stderr; on failure, read the error, fix and retry.",
            {
                "command": {"type": "string", "description": "The bash command line to execute."},
                "timeout": {"type": "integer", "description": f"Optional timeout in seconds (max {int(timeout_limit)})."},
            },
            ["command"],
        ),
        _tool_param(
            "read_file", "Read a file from the sandbox workspace (exact round-trip, prefer this over `cat` when you need file contents verbatim).",
            {"path": {"type": "string", "description": "File path relative to the workspace root."}}, ["path"],
        ),
        _tool_param(
            "write_file", "Write a file into the sandbox workspace (exact round-trip, prefer this over shell redirection for precise contents).",
            {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            ["path", "content"],
        ),
    ]


def _format_execution_result(result: ExecutionResult) -> str:
    ''' ExecutionResult → tool message content (截断统一在入对话处, 见 _FEEDBACK_CHAR_LIMIT) '''
    parts = [f"exit_code: {result.exit_code}"]
    if result.timed_out:
        parts.append("timed_out: True")
    if result.stdout:
        parts.append("stdout:\n" + result.stdout)
    if result.stderr:
        parts.append("stderr:\n" + result.stderr)
    return "\n".join(parts)


_GENERIC_AGENT_INSTRUCTION = (
    "\nYou are working autonomously in a sandbox: an isolated filesystem (HOME=/workspace, writable, persists across your tool calls) "
    "with optional network access. Get things done with the bash tool (run `ls-commands` to discover available commands); "
    "use read_file/write_file for exact file round-trips, and google_search when a web search helps. "
    "Command failures return exit_code/stdout/stderr — read the error, fix and retry. "
    "When the task is complete, submit your final result by calling the return_result tool.\n"
)


class GenericAgent(FunctionAgent):
    ''' 通用任务 agent: 模型在 bwrap 沙盒 (独立 fs + 可选网络) 内自主使用 bash/文件/搜索工具完成任务.

    两种用法:
    - 直接使用 (@ai_powered(agent=GenericAgent)): 空沙盒 + 默认配置 (timeout 30s, share_net True)
    - 继承覆盖 ClassVar: 预置 workspace 文件 (workspace_files) / 放宽超时 (bash_timeout_limit) / 关网 (share_net=False)

    沙盒生命周期 = 单次 run: run 开始创建 (跨多次工具调用共享), finally 清理, run 间不保留.
    工具执行错误 (含超时/路径越界) 一律文本化回喂模型自修复 (环内修复, 不烧 retry 预算);
    bwrap 后端缺失在构造期 (= 装饰期) fail-fast.

    交卷可靠性 (issue #28): 连续 stall_threshold 轮纯文本/空响应 → 注入交卷提醒 (单轮纯文本放任 —— 可能是规划);
    水位轴指数收缩提醒 (r ≤ 上次提醒剩余/2 触发) + r=1 强制 tool_choice 终点; return_result 提交环内校验回喂
    (结构化提交路径补齐 "工具错误一律环内修复" 的设计哲学); 纯文本整体恰为合法 Result[R] JSON 时直接收编.
    '''

    max_rounds: int = 16
    required_features: ClassVar[set[ModelFeature]] = {ModelFeature.tools}
    bash_timeout_limit: ClassVar[float] = 30.0  #: bash 硬上限 = 缺省值: 模型 per-call 可指定更短, clamp 到此
    share_net: ClassVar[bool] = True
    workspace_files: ClassVar[Mapping[str, str | bytes]] = {}
    #: 连续失速轮 (纯文本/空响应) 的交卷提醒阈值; 2 = 首轮纯文本放任 (可能是规划), 连续两轮只说不干判失速
    stall_threshold: ClassVar[int | None] = 2

    def __init__(self, spec: FunctionSpec, model_config: ModelConfig, connection: LlmConnection) -> None:
        super().__init__(spec, model_config, connection)
        if not self._backend_ready():
            raise ValueError(
                f"{type(self).__name__} requires the bubblewrap (bwrap) sandbox backend; "
                "install bubblewrap (Linux only) to use it"
            )

    @staticmethod
    def _backend_ready() -> bool:
        ''' 后端探测 (独立方法: 测试子类可覆盖以离线驱动) '''
        return shutil.which("bwrap") is not None

    def _create_sandbox(self) -> Sandbox:
        ''' 沙盒工厂 (独立方法: 测试子类可注入 Fake; run 期才创建, 与构造期探测分离) '''
        return Sandbox(share_net = type(self).share_net, workspace_files = type(self).workspace_files)

    @sync_compatible
    async def _execute_tool(self, tool_call: ChatCompletionMessageFunctionToolCall, sandbox: Sandbox) -> str:
        ''' 执行一次工具调用, 返回回喂文本; 一切失败都文本化 (模型自修复的素材), 不炸 run

        NOTE: 必须 @sync_compatible —— 它被 run 的 sync 转写体 await (easy-sync 约束, 裸 async def 会爆
        "coroutine has no attribute wait"); 其内 await 的 run_bash/_run_google_search 均为双实现 Waitable
        '''
        try:
            args = msgspec.json.decode(tool_call.function.arguments, type = dict[str, Any])
            name = tool_call.function.name
            if name == "bash":
                timeout = args.get("timeout", type(self).bash_timeout_limit)
                if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
                    timeout = type(self).bash_timeout_limit
                result = await sandbox.run_bash(args["command"], min(float(timeout), type(self).bash_timeout_limit))
                return _format_execution_result(result)
            if name == "read_file":
                return sandbox.read_file(args["path"]).decode("utf-8", errors = "replace")
            if name == "write_file":
                sandbox.write_file(args["path"], args["content"])
                return f"written: {args['path']}"
            if name == "google_search":
                return await _run_google_search(**args)
        except Exception as e:  # 工具失败 (参数缺失/路径越界/执行异常) → 文本回喂
            return f"{type(e).__name__}: {e}"
        return f"unknown tool: {tool_call.function.name}"

    @sync_compatible
    async def run(self, arguments_json: str) -> str:
        # return_schema 为 raw 形态, 工具参数 schema 按通道能力门控展开/透传 (issue 18);
        # 工具面是纯计算 (不依赖 sandbox), 必须置于 _create_sandbox 之前 —— resolve_schema 可抛
        # RecursiveSchemaError, 先于 mkdtemp fail-fast 方能保证 sandbox 创建后的任何路径都在 try/finally 内
        refs_allowed = ModelFeature.schema_refs in self.model_config.supported_features
        tools = _generic_agent_tools(type(self).bash_timeout_limit) + [return_result_tool_param(resolve_schema(self.spec.return_schema, refs_allowed = refs_allowed))]
        if os.getenv("GOOGLE_API_KEY"):  # 搜索是便利件而非依赖件: 无 key 静默不注册 (bash 里还有 curl)
            tools.append(google_search.schema(refs_allowed = refs_allowed))
        sandbox = self._create_sandbox()
        try:
            system_prompt = base_system_prompt(self.spec) + _GENERIC_AGENT_INSTRUCTION
            conversation: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arguments_json},
            ]
            watchdog = _SubmissionWatchdog(self.max_rounds, type(self).stall_threshold, _WORK_THEN_SUBMIT_REMINDER)

            for _round in range(self.max_rounds):
                remaining = self.max_rounds - _round
                request_kwargs: dict[str, Any] = {"model": self.model_config.model_name, "messages": conversation, "tools": tools}
                if remaining == 1:  # 强制终点 (issue #28); 不支持 tool_choice 的通道忽略之, 由最后通牒措辞兜底
                    request_kwargs["tool_choice"] = _FORCED_RETURN_RESULT
                response = await self.connection.chat_completions(**request_kwargs)

                message = response.choices[0].message
                event_reminders: list[str] = []

                tool_calls = message.tool_calls
                if tool_calls is not None:
                    fn_tool_calls = [t for t in tool_calls if isinstance(t, ChatCompletionMessageFunctionToolCall)]
                    if len(fn_tool_calls) != len(tool_calls):  # custom tool call 等变体: 拒绝 (避免 "有 tool_calls 无 tool 响应" 的非法对话序列)
                        raise AgentRunError(f"{type(self).__name__} got unexpected tool call(s): {tool_calls}")
                    submission, submission_errors = _scan_submission(fn_tool_calls, self._result_type)
                    if submission is not None:  # 合法提交在批内: 其余调用无执行必要 (结果无人消费, run 即将结束)
                        return submission
                    watchdog.note_tool_round()
                    conversation.append(cast(ChatCompletionMessageParam, message.to_dict()))  # assistant 工具调用轮入栈
                    for tool_call in fn_tool_calls:  # 并行协议: 同批每个 tool_call 均须有 tool 响应 (issue #28), 按原始调用顺序回喂
                        content = submission_errors.get(tool_call.id) if tool_call.function.name == "return_result" else None
                        if content is None:  # 普通工具: 执行结果回喂 → 模型据 stdout/stderr 自修复
                            content = (await self._execute_tool(tool_call, sandbox))[-_FEEDBACK_CHAR_LIMIT:]
                        conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

                elif not message.content:  # 空响应且无工具调用: 无任何规划语义 → 立即追问 (阈值天然为 1); 计入失速轮
                    conversation.append({"role": "assistant", "content": ""})  # 保留 assistant 占位以维持 role 交替惯例
                    event_reminders.append(watchdog.note_empty_round())

                else:
                    incorporated = _try_incorporate(message.content, self._result_type)
                    if incorporated is not None:
                        return incorporated
                    conversation.append(cast(ChatCompletionMessageParam, message.to_dict()))  # 中间推理保留在对话中 (与 ChatBot 同款窄化惯例)
                    reminder = watchdog.note_stall_round()
                    if reminder is not None:
                        event_reminders.append(reminder)

                injection = watchdog.round_end_injection(remaining - 1, event_reminders)
                if injection is not None:
                    conversation.append({"role": "user", "content": injection})

            raise AgentRunError(f"{type(self).__name__} did not submit a result within {self.max_rounds} rounds")
        finally:
            sandbox.cleanup()
