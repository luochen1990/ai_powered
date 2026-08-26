# 用例: 有序数组二分查找索引 — 多参数复合推理
# 溯源: ai_powered@test/examples/ai_powered_decorator/binary_search.py@a1a39e0
# 判定: 等值 (present: 期望下标; absent: 期望 -1)

import random

from ai_powered.llm_bench import Instance, case


@case(
    id="func-simu/binary-search",
    difficulty="medium",
    source="ai_powered:test/examples/ai_powered_decorator/binary_search.py@a1a39e0",
)
def binary_search(arr: list[int], x: int) -> int:
    """Returns the index of x in arr if present, else -1, the index starts from 0
    e.g.
    binary_search([10, 101, 1002, 10003], 10) == 0
    binary_search([10, 101, 1002, 10003], 10003) == 3
    """
    ...


def _one(rng: random.Random) -> Instance:
    # 升序唯一数组; 六成命中 (期望下标), 四成落空 (期望 -1)
    arr = sorted(rng.sample(range(0, 1000), rng.randint(4, 12)))
    if rng.random() < 0.6:
        idx = rng.randrange(len(arr))
        return Instance(args={"arr": arr, "x": arr[idx]}, expected=idx)
    absent = rng.choice([x for x in range(-5, 1005) if x not in set(arr)])
    return Instance(args={"arr": arr, "x": absent}, expected=-1)


def gen(rng: random.Random, n: int) -> list[Instance]:
    return [_one(rng) for _ in range(n)]
