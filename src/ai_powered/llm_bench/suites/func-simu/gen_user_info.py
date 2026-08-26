# 用例: 随机生成用户信息 — 无输入生成型任务 (属性判定)
# 溯源: ai_powered@test/examples/ai_powered_decorator/gen_user_info.py@a1a39e0
# 偏离上游: country 由 Optional[str] 收紧为 Optional[Country] (Literal 枚举) — bench 侧有意强化,
#           目的是测"模型对 schema 中显式 enum 约束的遵循能力"; 枚举随 return schema 进入请求, 模型可见
# 判定分工: 枚举强制单点在 schema 层 (msgspec 按 Literal 解码, 枚举外取值 -> schema_violation);
#           judge 只管结构 (字段类型/非空), 不重复查枚举 (解码成功 ⟹ 取值必在枚举内, 再查即不可达代码)
# 路径敏感性: structured_outputs 路径下 provider 做约束解码, 枚举违反几乎不可能发生 —
#           该用例主要在 tools/chat 路径下测模型的枚举遵循
# 说明: None 字段语义 ("None is used for unknown fields") 沿用上游注解约定

import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

from ai_powered.llm_bench import Instance, case

Country = Literal["China", "USA", "Russia", "UK", "France"]  # 五常


@dataclass
class UserInfo:
    name: str
    country: Optional[Country]
    age: Optional[str]


@case(
    id="func-simu/gen-user-info",
    difficulty="easy",
    source="ai_powered:test/examples/ai_powered_decorator/gen_user_info.py@a1a39e0 (country 收紧为 Literal, 见头注)",
)
def gen_user_info() -> UserInfo:
    """ramdomly generate user info for testing, None is used for unknown fields"""
    ...


def gen(rng: random.Random, n: int) -> list[Instance]:
    # 无输入任务: 实例仅承载重复触发 (每次调用应产出不同的合法 UserInfo)
    return [Instance(args={}) for _ in range(n)]


def judge(instance: Instance, output: Any) -> bool:
    """属性判定: 结构合法 (字段类型/非空); 枚举遵循由 schema 解码层强制, 见头注"""
    if not isinstance(output, UserInfo):
        return False
    if not isinstance(output.name, str) or not output.name.strip():
        return False
    if output.country is not None and (not isinstance(output.country, str) or not output.country.strip()):
        return False
    if output.age is not None and (not isinstance(output.age, str) or not output.age.strip()):
        return False
    return True
