'''
验证 deref 能正确展开 list[NestedStruct] 等"容器型嵌套"中残留的 $ref.

背景: 旧版 _deref 只递归 properties, 不递归 items / additionalProperties / prefixItems / anyOf 等容器键,
导致 list[Inner] 生成的 {"type":"array","items":{"$ref":"#/$defs/Inner"}} 中 $ref 残留,
而 $defs 又被上层 Result 包装丢弃, 最终 OpenAI/ollama 报 "Error resolving ref".

读者: 任何修改 schema_deref.py 的人 (回归门禁).
'''
import json
from typing import Any, Optional, Generic, TypeVar

import msgspec

import importlib.util
import os

# 直接按文件加载, 避免触发 ai_powered/__init__.py 的重依赖 (openai/easy_sync 等),
# 使本脚本在最小环境 (仅需 msgspec) 下即可运行.
_spec = importlib.util.spec_from_file_location(
    "schema_deref",
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "ai_powered", "schema_deref.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
deref = _mod.deref


class Inner(msgspec.Struct):
    x: int
    y: str


class Outer(msgspec.Struct):
    items: list[Inner]
    count: int


T = TypeVar("T")


class Result(msgspec.Struct, Generic[T]):
    result: T


def has_ref(d: Any) -> bool:
    '''递归检查 schema 树中是否还有 $ref 字段.'''
    if isinstance(d, dict):
        return '$ref' in d or any(has_ref(v) for v in d.values())
    if isinstance(d, list):
        return any(has_ref(i) for i in d)
    return False


def check_case(name: str, typ: type) -> None:
    schema = msgspec.json.schema(typ)
    assert has_ref(schema), f"{name}: 测试用例本身的 schema 应包含 $ref, 否则测试无效"
    derefed = deref(schema)
    assert not has_ref(derefed), f"{name}: deref 结果仍有 $ref 残留:\n{json.dumps(derefed, indent=2)}"
    print(f"PASSED: {name} -> $ref 全部展开")


if __name__ == "__main__":
    # 核心场景: list[NestedStruct] (micro-loop Review.issues 的同款形状)
    check_case("Result[Outer] (含 list[Inner])", Result[Outer])

    # 直接 list[Inner] (无外层 Result 包装, $defs 在最外层)
    check_case("list[Inner]", list[Inner])

    # 直接 Inner (顶层 $ref 形式)
    check_case("Outer (直接结构)", Outer)

    # dict[str, Inner] -> additionalProperties
    check_case("dict[str, Inner]", dict[str, Inner])

    # Optional[Inner] -> anyOf
    class WithOpt(msgspec.Struct):
        v: Optional[Inner]
    check_case("Optional[Inner] (anyOf)", WithOpt)

    # tuple[Inner, Inner] -> prefixItems + items:false
    class WithTuple(msgspec.Struct):
        t: tuple[Inner, Inner]
    check_case("tuple[Inner, Inner] (prefixItems)", WithTuple)

    print("\n全部 PASSED")
