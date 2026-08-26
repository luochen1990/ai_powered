# 用例: 两整数相加 — 最基础的类型化数值推理
# 溯源: ai_powered@test/examples/ai_powered_decorator/add.py@a1a39e0
# 判定: 等值 (expected = a + b)

import random

from ai_powered.llm_bench import Instance, case


@case(id="func-simu/add", difficulty="easy", source="ai_powered:test/examples/ai_powered_decorator/add.py@a1a39e0")
def add(a: int, b: int) -> int:
    """Compute the sum of the two given integers"""
    ...


def gen(rng: random.Random, n: int) -> list[Instance]:
    # 边界实例 (确定性) + 随机实例 (覆盖典型量级, 含负数)
    edges = [(0, 0), (-1, 1), (10**12, 10**12), (-(10**12), 10**12)]
    pairs = edges[: min(len(edges), n)] + [(rng.randint(-(10**6), 10**6), rng.randint(-(10**6), 10**6)) for _ in range(max(0, n - len(edges)))]
    return [Instance(args={"a": a, "b": b}, expected=a + b) for a, b in pairs]
