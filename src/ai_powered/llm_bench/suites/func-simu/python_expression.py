# 用例: 数学表达式转 Python 表达式 — 语义保持的语法转换 (可执行判定)
# 溯源: ai_powered@test/examples/ai_powered_decorator/calculator.py@a1a39e0 (python_expression_generator)
# 判定: 可执行 —— 受限沙箱 eval 模型输出的 Python 表达式, 结果值须等于构造时已知的真值
# 任务面: 输入含非 Python 语法 (^ 幂, sum_from_to(a,b) 即 sum(range(a,b+1))), 模型须转为合法 Python
# 已知水位差异 (迁移时显式标注, # 注释不参与 digest): 本文件的 _safe_eval 为 bench 侧原版,
#   字符闸门挡属性链逃逸但无 DoS 静态缓解 (如 9**9**9**9 资源耗尽), 弱于本仓
#   ai_powered.utils.safe_eval 的水位 (幂指数静态折叠/位数上界); 刻意不复用后者 —— suite 内容
#   与 bench 侧逐字节等价是迁移契约, 修复须 bench 侧先行 (身份断代协调), 见 ai-powered#12

import random
from typing import Any

from ai_powered.llm_bench import Instance, case

# 受限 eval 白名单 (仅表达式求值所需内置函数)
_EVAL_BUILTINS: dict[str, Any] = {"sum": sum, "range": range, "min": min, "max": max, "abs": abs, "round": round, "pow": pow}


def _safe_eval(expr: str) -> int:
    """受限表达式求值: 禁属性访问(".")与下划线名("__"), 清空 __builtins__ (表达式逃逸需属性链)"""
    if len(expr) > 200 or "." in expr or "_" in expr:
        raise ValueError(f"unsafe or oversized expression: {expr!r}")
    return eval(expr, {"__builtins__": {}}, _EVAL_BUILTINS)  # noqa: S307 — 输入已过白名单字符闸门


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
        return _safe_eval(output) == instance.expected
    except Exception:  # noqa: BLE001 — 任何求值失败均为不通过
        return False
