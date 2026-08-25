AI Powered
==========

[![codecov](https://codecov.io/github/luochen1990/ai_powered/graph/badge.svg?token=OBG1BWIKC2)](https://codecov.io/github/luochen1990/ai_powered)

动机
---

当前 AI (这里主要指大语言模型) 已经发展到了一个相当实用的阶段，但仍然有很多传统的软件并未从中受益，该项目旨在为各类软件运用 AI 能力提供一些便捷的工具。

通过这些工具，你不必为了利用 AI 能力而重新设计整个软件，甚至不需要增加任何用户可见的功能模块，只要你发现你项目中的某个函数所实现的功能有可能被AI解答得更好，你就可以用AI来替换它的实现。

用法
---

安装只需要通过 `pip install ai_powered` 或 `poetry add ai_powered`.

它提供了以下工具:

### `@ai_powered` 装饰器

这个装饰器为你的函数签名赋予一个AI实现, 它会读取 doc string 并调用 LLM 来获取函数的返回值

```python
from ai_powered import ai_powered

@ai_powered
def get_python_expression(expr: str) -> str:
    ''' 将用户输入的数学表达式转换成合法的python表达式 '''
    ...
```

你也可以在参数和返回值中使用更复杂的数据结构，但请确保它们有完整的类型标注

```python
@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[int]

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    '''
    从这段自我介绍中提取出用户信息
    '''
```

更多例子参见 [这里](/test/examples/ai_powered_decorator/)

装饰器也接受关键字参数 (两种形式都可用于 sync 和 async 函数):

```python
@ai_powered(retry = 2)
async def add(a: int, b: int) -> int:
    ...
```

`retry = N` 表示遇到瞬时失败 (网络/API 错误、响应非法或不符合 schema、或响应缺少期望结构) 时最多额外重试 N 次; 默认 `retry = 0` 保持旧的快速失败行为. 注意本层重试与 OpenAI SDK 自身的 `max_retries` (网络错误) 相互独立、可能叠加. 重试耗尽后原样抛出最后一个异常.

还可以用 `provider/model` 前缀名在函数粒度选择模型 (内置注册表解析, 无需切换环境变量). 可注册自定义 provider, 注意注册必须发生在被装饰函数定义**之前**:

```python
from ai_powered import ai_powered, register_model

register_model("mygateway/my-model", base_url = "https://llm.example.com/v1", api_key_env = "MYGATEWAY_API_KEY")

@ai_powered(model = "deepseek/deepseek-chat")
def translate(text: str) -> str:
    ''' 将文本翻译为英文 '''
    ...
```

解析规则与内置 provider 列表参见 [Configuration](/doc/en/Configuration.md).

还可以用自定义 agent (FunctionAgent 子类) 替换默认的单次请求模拟, 例如内置的多轮推理引擎:

```python
from ai_powered import ReasoningAgent, ai_powered

@ai_powered(agent = ReasoningAgent, retry = 1)
def solve_word_problem(problem: str) -> int:
    ''' 分步求解应用题, 返回数字答案 '''
    ...
```

agent 在装饰期为每个函数构造一次, 收到 (spec, model_config, connection); 返回值的类型校验与 retry 依然生效. 子类契约见 `FunctionAgent` docstring.

参数设计的决策记录参见 [这里](/doc/en/DecoratorParams.md).

### `@make_tool` 装饰器

这个装饰器将普通Python函数转换为 LLM 的 [function calling](https://platform.openai.com/docs/guides/function-calling) 功能可以使用的 tool

```python
from ai_powered import make_tool
from ai_powered.utils.safe_eval import safe_eval
import openai

@make_tool
def calculator(python_expression: str) -> str:
    ''' 求值python表达式(仅支持内置函数), 可以用来解决数学问题 '''
    return safe_eval(python_expression)

client = openai.OpenAI()
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = self.conversation,
    tools = [ calculator.schema() ],
    tool_choice = calculator.choice()
)
```

### `ChatBot` 类

这个类实现了一个AI聊天机器人，你可以继承它实现自己的ChatBot子类，以指定要用的系统提示词和工具，它会帮你处理这些复杂的工具调用过程

```python
class MyChatBot (ChatBot):
    system_prompt = '''
    Please answer the user's questions. If any calculations are required, use the calculator available in the tool. It supports complex Python expressions. When using it, make sure to convert the user's mathematical expression to a valid Python expression. Do not use any undefined functions; if the user's expression includes function calls, convert them to Python's built-in functions or syntax.
    '''
    tools = (calculator,)

if __name__ == "__main__":
    bot = MyChatBot()
    print(bot.chat('hello, please tell me the result of 2^10 + 3^4'))
    print(bot.chat('and what is above result divided by 2?'))
    print(f"{bot.conversation =}")
```

更多例子参见 [这里](/test/examples/chat_bot/)

当前限制和未来规划
---------------

- 目前仅提供了Python实现，但事实上这个模式可以复制到能运行时获取类型标注的任何其他语言
- 目前 OpenAI 的 function calling 功能并不能识别 schema 中的引用，所以当类型复杂到会产生引用时将可能出现问题，这可以通过 deref 来缓解，但当遇到递归类型时就不得不保留 ref，所以这个问题最终可能要等 LLM 提供商在训练数据集中加上这类数据后才能真正解决
- 当前 LLM 产生的数据并不能百分百遵循所提供的 JSON Schema，重试机制只能缓解偶发性偏差 (已支持, 见上文 retry 参数); 且重试只能覆盖结构性失败, "合法但语义错误"的结果无法被检测 (没有标准答案可对照)

关于贡献代码
----------

1. 该项目会一直开源，但我暂时没有想好应该使用哪个开源协议，如果你不介意这一点那么可以直接PR，否则可能我们需要先讨论一下协议的事情
2. 目前所有代码全部启动 Pyright 严格模式类型检查，如果有类型错误会被 github actions 挡住，同时我们也不建议在非必要的情况下使用 `Any` 或 `#type: ignore`
3. 测试覆盖率会被持续监测，建议永远为你的代码提供测试，甚至在编码前就先准备好测试
4. 关于开发环境，建议安装 `nix` 和 `direnv` 这样你将自动获得一个可用的开发环境，当然 `poetry shell` 也是不错的选择（如果你已经在使用 poetry 的话）
