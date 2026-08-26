# 用例: 数学表达式转 Python 表达式 — 语义保持的语法转换 (可执行判定)
# 溯源: ai_powered@test/examples/ai_powered_decorator/calculator.py@a1a39e0 (python_expression_generator)
# 判定: 可执行 —— 受限沙箱 eval 模型输出的 Python 表达式, 结果值须等于构造时已知的真值
# 任务面: 输入含非 Python 语法 (^ 幂, sum_from_to(a,b) 即 sum(range(a,b+1))), 模型须转为合法 Python
# 沙箱水位 (llm-bench#12 上游半): 判定器委托 ai_powered.utils.safe_eval (issue #10 修复产物) —
#   AST 允许清单门 (fail-closed) + 幂/移位/序列的静态 DoS 缓解 (如 9**9**9**9 资源耗尽被拒)。
#   取代迁移期的字符闸门原版 (无 DoS 缓解); 原版刻意的 "与 bench 侧逐字节等价" 约束已由分层
#   digest 的判定段语义取代 — judge 修复 = 同题库不同判定, bench 侧 bump 依赖 rev 后重 pin 即吸收
# 已知残余: ** 算子操作数含动态量时静态不可判界、无运行时守卫可包装 (如动态底数 + 可折叠
#   有界指数的组合, utils 头注文档化边界, 与旧实现暴露面相同); 本 judge 为进程内直调,
#   无 bwrap 兜底档 —— 进程级隔离属后续 backlog

import random
from typing import Any

from ai_powered.llm_bench import Instance, case
from ai_powered.utils.safe_eval import safe_eval


@case(
    id="func-simu/python-expression",
    difficulty="medium",
    source="ai_powered:test/examples/ai_powered_decorator/calculator.py@a1a39e0",
)
def python_expression_generator(expr: str) -> str:
    """将用户输入的数学表达式转换成合法的python表达式, 注意不要使用任何未定义的函数，如果用户表达式中有类似函数调用的表达，请转换为python内置函数或语法"""
    ...


def _leaf(rng: random.Random) -> tuple[str, int]:
    value = rng.randint(0, 20)
    return str(value), value


def _node(rng: random.Random, depth: int) -> tuple[str, int]:
    """递归构造表达式树, 返回 (表面文本, 精确真值); 二元运算保持值域有界"""
    if depth <= 0:
        return _leaf(rng)
    kind = rng.choice(["leaf", "add", "sub", "mul", "pow", "sum_from_to"])
    if kind == "leaf":
        return _leaf(rng)
    if kind == "add":
        a_text, a_val = _node(rng, depth - 1)
        b_text, b_val = _node(rng, depth - 1)
        return f"({a_text} + {b_text})", a_val + b_val
    if kind == "sub":
        a_text, a_val = _node(rng, depth - 1)
        b_text, b_val = _leaf(rng)
        return f"({a_text} - {b_text})", a_val - b_val
    if kind == "mul":
        a_text, a_val = _node(rng, depth - 1)
        b_val = rng.randint(0, 5)
        return f"({a_text} * {b_val})", a_val * b_val
    if kind == "pow":
        base = rng.randint(1, 5)
        exp = rng.randint(0, 3)
        return f"{base}^{exp}", base**exp  # 表面用 ^ (非 Python 语法)
    # sum_from_to: 表面非 Python 函数, 真值为闭区间求和
    a = rng.randint(0, 8)
    b = a + rng.randint(0, 6)
    return f"sum_from_to({a}, {b})", sum(range(a, b + 1))


def gen(rng: random.Random, n: int) -> list[Instance]:
    instances = []
    for _ in range(n):
        text, value = _node(rng, 3)
        instances.append(Instance(args={"expr": text}, expected=value))
    return instances


def judge(instance: Instance, output: Any) -> bool:
    """沙箱执行模型输出的 Python 表达式, 与构造真值比对"""
    if not isinstance(output, str):
        return False
    try:
        return safe_eval(output) == instance.expected
    except Exception:  # noqa: BLE001 — 任何求值失败均为不通过
        return False
