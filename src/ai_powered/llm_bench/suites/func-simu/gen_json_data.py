# 用例: 按深度生成 JSON 数据 — 结构约束满足型任务 (属性判定)
# 溯源: ai_powered@test/examples/ai_powered_decorator/gen_json_data.py@a1a39e0
# 偏离上游: 上游返回 Any (JSON Schema 为空 {}, 零约束力 — 字符串化 JSON 也是合法 Any, 解码层无从拒绝);
#           本 bench 收紧为 JsonTree = Union[dict, list], schema 携带 object|array 约束 (模型可见),
#           字符串化行为在 msgspec 解码层被拒绝 (归因 schema_violation 而非 judge 兜底的 wrong_value)
# 判定: 属性 —— 嵌套深度 >= 请求深度, 规模受控 (防爆内存);
#       schema 管形状, decoder 管强制, judge 管性质

import random
from typing import Any, Union

from ai_powered.llm_bench import Instance, case

JsonTree = Union[dict[str, Any], list[Any]]


@case(
    id="func-simu/gen-json-data",
    difficulty="medium",
    source="ai_powered:test/examples/ai_powered_decorator/gen_json_data.py@a1a39e0 (返回类型 Any 收紧为 JsonTree, 见头注)",
)
def gen_json_data(depth: int) -> JsonTree:
    """random generate some valid json data for test usage"""
    ...


def _depth(value: Any) -> int:
    """标量深度为 1, 容器为其子项最大深度 + 1"""
    if isinstance(value, list):
        return 1 + max((_depth(v) for v in value), default=0)
    if isinstance(value, dict):
        return 1 + max((_depth(v) for v in value.values()), default=0)
    return 1


def _node_count(value: Any) -> int:
    if isinstance(value, list):
        return 1 + sum(_node_count(v) for v in value)
    if isinstance(value, dict):
        return 1 + sum(_node_count(v) for v in value.values())
    return 1


def gen(rng: random.Random, n: int) -> list[Instance]:
    return [Instance(args={"depth": rng.randint(2, 6)}) for _ in range(n)]


def judge(instance: Instance, output: Any) -> bool:
    """属性判定: 合法 JSON 树 / 深度达标 / 规模受控"""
    if _node_count(output) > 5000:
        return False
    return _depth(output) >= instance.args["depth"]
