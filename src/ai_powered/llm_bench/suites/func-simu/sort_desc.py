# 用例: 列表降序排序 — 结构化输出 (list[int]) 的保序保元测试
# 溯源: ai_powered@test/examples/ai_powered_decorator/sort_desc.py@a1a39e0
# 判定: 等值 (expected = sorted(xs, reverse=True); 含重复元素场景)

import random

from ai_powered.llm_bench import Instance, case


@case(id="func-simu/sort-desc", difficulty="easy", source="ai_powered:test/examples/ai_powered_decorator/sort_desc.py@a1a39e0")
def sort(xs: list[int]) -> list[int]:
    """sort elements in decreasing order"""
    ...


def _one(rng: random.Random) -> Instance:
    # 值域收窄使重复元素必然出现 (重复元素的稳定排序是真实难点)
    xs = [rng.randint(-20, 20) for _ in range(rng.randint(2, 12))]
    return Instance(args={"xs": xs}, expected=sorted(xs, reverse=True))


def gen(rng: random.Random, n: int) -> list[Instance]:
    return [_one(rng) for _ in range(n)]
