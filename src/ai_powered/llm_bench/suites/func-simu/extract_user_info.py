# 用例: 自我介绍文本抽取结构化用户信息 — 自然语言到类型化 dataclass 的提取
# 溯源: ai_powered@test/examples/ai_powered_decorator/extract_user_info.py@a1a39e0
# 判定: 等值 (generator 模板化生成文本与配套 UserInfo 真值; 国家用无歧义标准短名, 严格匹配)
# 说明: 原测试的 is_same_country 模糊判定依赖额外 LLM 调用, 不符合本 bench 纯程序化判定原则,
#       改为控制生成侧使真值无歧义 (相对比较的严格性对所有模型一致)

import random
from dataclasses import dataclass
from typing import Optional

from ai_powered.llm_bench import Instance, case

# 生成侧数据池 (标准短名, 模型无需归一化即可命中的写法)
_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo"]
_COUNTRIES = ["China", "France", "Germany", "Japan", "Brazil", "India", "Canada", "Italy", "Russia", "Australia"]
_TEMPLATES = [
    "Hello, my name is {name}, and I am from {country}. I am {age} years old",
    "Hi! I'm {name}, a {age}-year-old from {country}.",
    "I am {name}. I was born in {country} and I am {age} years old now.",
    "Hey there, {name} here — {age} years old, living in {country}.",
]


@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[int]


@case(
    id="func-simu/extract-user-info",
    difficulty="medium",
    source="ai_powered:test/examples/ai_powered_decorator/extract_user_info.py@a1a39e0",
)
def extract_user_info(raw_text: str) -> UserInfo:
    """
    Obtain complete information as much as possible, unless it is truly not mentioned;
    when identifying countries, match them to the closest real existing country
    """
    ...


def gen(rng: random.Random, n: int) -> list[Instance]:
    instances = []
    for _ in range(n):
        name, country, age = rng.choice(_NAMES), rng.choice(_COUNTRIES), rng.randint(8, 70)
        text = rng.choice(_TEMPLATES).format(name=name, country=country, age=age)
        instances.append(Instance(args={"raw_text": text}, expected=UserInfo(name=name, country=country, age=age)))
    return instances
