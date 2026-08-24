'''
职责: FunctionAgent 协议 (函数执行引擎) + 参考实现 ReasoningAgent.

协议定位 (doc/en/DecoratorParams.md D3):
  - @ai_powered(agent=SomeAgent) 用指定 agent 类替换默认的 "单次 LLM 请求" 模拟器;
    多轮推理/工具使用/代码执行等都只是引擎的实现细节, 出口校验 (Result[R] 解码) 对 agent 一视同仁,
    agent 不能绕过被装饰函数的类型契约
  - agent 与 retry 正交: retry 重试的是整个 agent run (而非 agent 内部的单次 LLM 请求)
  - 沙盒 (执行 LLM 生成代码等) 按决策 D4 defer, 后续作为 agent 的属性增量开发

实现约定 (easy-sync 约束, 见 decorators.py 的 NOTE):
  - 子类的 run 必须以 @sync_compatible 装饰的 async def 形式实现 —— 它会被 sync 转写体经 .wait() 调用
'''
from abc import ABC, abstractmethod
from typing import ClassVar

from easy_sync import sync_compatible
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from ai_powered.constants import base_system_prompt
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import ModelConfig
from ai_powered.tool_call import return_result_tool_param


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
