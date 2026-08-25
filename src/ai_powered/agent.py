'''
职责: FunctionAgent 协议 (函数执行引擎) + 参考实现 ReasoningAgent.

协议定位 (doc/en/DecoratorParams.md D3):
  - @ai_powered(agent=SomeAgent) 用指定 agent 类替换默认的 "单次 LLM 请求" 模拟器;
    多轮推理/工具使用/代码执行等都只是引擎的实现细节, 出口校验 (Result[R] 解码) 对 agent 一视同仁,
    agent 不能绕过被装饰函数的类型契约
  - agent 与 retry 正交: retry 重试的是整个 agent run (而非 agent 内部的单次 LLM 请求)
  - 沙盒 (执行 LLM 生成代码) 已按 D4 落地 (B4): GenericAgent 内化为 agent 属性 (私有 sandbox.py 后端)

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
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ai_powered.constants import base_system_prompt
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import ModelConfig
from ai_powered.sandbox import COMMAND_GROUPS, ExecutionResult, Sandbox
from ai_powered.tool_call import return_result_tool_param
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

    def __init__(self, spec: FunctionSpec, model_config: ModelConfig, connection: LlmConnection) -> None:
        self.spec = spec
        self.model_config = model_config
        self.connection = connection

    @sync_compatible
    @abstractmethod
    async def run(self, arguments_json: str) -> str:
        ''' 子类必须以 @sync_compatible async def run 覆盖 (见类 docstring 的 easy-sync 约定) '''


class ReasoningAgent(FunctionAgent):
    ''' 参考实现: 多轮推理引擎 —— 模型在对话中自由思考, 通过 return_result 工具提交结果

    与默认单次模拟的差别: 模型可以先输出若干轮纯文本推理 (不急于作答), 再调用工具提交结构化结果;
    max_rounds 上限防失控. 适用于需要 step-by-step 推理的任务. (需要模型支持 function calling)
    '''

    max_rounds: int = 8
    required_features: ClassVar[set[ModelFeature]] = {ModelFeature.tools}

    @sync_compatible
    async def run(self, arguments_json: str) -> str:
        system_prompt = base_system_prompt(self.spec) + "\nThink step by step as you need, then submit your final result by calling the return_result tool.\n"

        conversation: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": arguments_json},
        ]

        for _round in range(self.max_rounds):
            response = await self.connection.chat_completions(
                model = self.model_config.model_name,
                messages = conversation,
                tools = [return_result_tool_param(self.spec.return_schema)],
            )

            message = response.choices[0].message

            tool_calls = message.tool_calls
            if tool_calls is not None:
                for tool_call in tool_calls:
                    if isinstance(tool_call, ChatCompletionMessageFunctionToolCall) and tool_call.function.name == "return_result":
                        return tool_call.function.arguments
                called = [t.function.name for t in tool_calls if isinstance(t, ChatCompletionMessageFunctionToolCall)]
                raise AgentRunError(f"ReasoningAgent got unexpected tool call(s): {called or tool_calls}")

            if not message.content:  # 空响应且无工具调用, 追问 (保留 assistant 占位以维持 role 交替惯例)
                conversation.append({"role": "assistant", "content": ""})
                conversation.append({"role": "user", "content": "Please call the return_result tool to submit your result."})
                continue

            conversation.append(message.to_dict())  #type: ignore[arg-type]  # assistant 中间推理保留在对话中 (与 ChatBot 同款窄化惯例)

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
    '''

    max_rounds: int = 16
    required_features: ClassVar[set[ModelFeature]] = {ModelFeature.tools}
    bash_timeout_limit: ClassVar[float] = 30.0  #: bash 硬上限 = 缺省值: 模型 per-call 可指定更短, clamp 到此
    share_net: ClassVar[bool] = True
    workspace_files: ClassVar[Mapping[str, str | bytes]] = {}

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
        sandbox = self._create_sandbox()
        tools = _generic_agent_tools(type(self).bash_timeout_limit) + [return_result_tool_param(self.spec.return_schema)]
        if os.getenv("GOOGLE_API_KEY"):  # 搜索是便利件而非依赖件: 无 key 静默不注册 (bash 里还有 curl)
            tools.append(google_search.schema())
        try:
            system_prompt = base_system_prompt(self.spec) + _GENERIC_AGENT_INSTRUCTION
            conversation: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arguments_json},
            ]

            for _round in range(self.max_rounds):
                response = await self.connection.chat_completions(
                    model = self.model_config.model_name,
                    messages = conversation,
                    tools = tools,
                )

                message = response.choices[0].message
                tool_calls = message.tool_calls
                if tool_calls is not None:
                    fn_tool_calls = [t for t in tool_calls if isinstance(t, ChatCompletionMessageFunctionToolCall)]
                    if len(fn_tool_calls) != len(tool_calls):  # custom tool call 等变体: 拒绝 (避免 "有 tool_calls 无 tool 响应" 的非法对话序列)
                        raise AgentRunError(f"{type(self).__name__} got unexpected tool call(s): {tool_calls}")
                    submission: str | None = None
                    tool_results: list[ChatCompletionToolMessageParam] = []
                    for tool_call in fn_tool_calls:
                        if tool_call.function.name == "return_result":
                            submission = tool_call.function.arguments
                        else:
                            tool_results.append({
                                "role": "tool", "tool_call_id": tool_call.id,
                                "content": (await self._execute_tool(tool_call, sandbox))[-_FEEDBACK_CHAR_LIMIT:],
                            })
                    if submission is not None:  # 与 ReasoningAgent 同惯例: 并行调用中取 return_result, 忽略其余
                        return submission
                    conversation.append(cast(ChatCompletionMessageParam, message.to_dict()))  # assistant 工具调用轮入栈
                    conversation.extend(tool_results)  # 执行结果回喂 → 模型据 stdout/stderr 自修复
                    continue

                if not message.content:  # 空响应且无工具调用, 追问 (保留 assistant 占位以维持 role 交替惯例)
                    conversation.append({"role": "assistant", "content": ""})
                    conversation.append({"role": "user", "content": "Please use the sandbox tools to work on the task, then call return_result to submit your result."})
                    continue

                conversation.append(cast(ChatCompletionMessageParam, message.to_dict()))  # 中间推理保留在对话中 (与 ChatBot 同款窄化惯例)

            raise AgentRunError(f"{type(self).__name__} did not submit a result within {self.max_rounds} rounds")
        finally:
            sandbox.cleanup()
