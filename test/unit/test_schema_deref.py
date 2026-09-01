'''
职责: schema_deref 模块的离线单元测试 —— $ref 展开 (deref) / 递归环检测 (RecursiveSchemaError) /
能力门控 schema 变体选择 (resolve_schema) / $defs 根提升 (wrap_result_schema / shallow_expand).

issue 18 阶段 0+1 的核心不变式:
  - 非递归 schema 恒走 deref, 输出与历史行为零语义漂移 (结果中无 $ref / 无 $defs;
    容器顶层死 $defs 键按契约剥除 —— master 请求曾携带, 字节级差异、语义等价)
  - 递归 schema: refs_allowed → 原样透传 raw ($ref/$defs 保留); 否则 RecursiveSchemaError fail-fast (取代裸 RecursionError)
  - 包装/浅展开时 $defs 必须随 $ref 一起留在文档根 (JSON Schema fragment 从文档根解析, 不提升即悬空)
'''
import json
from typing import List, Optional

import msgspec
import pytest

from ai_powered.schema_deref import RecursiveSchemaError, deref, resolve_schema, shallow_expand, wrap_result_schema


class ArgObj(msgspec.Struct):
    # 非递归参数对象 (msgspec 对 Struct 的 schema 顶层为 $ref 形态); 不写 docstring 以免 msgspec 把它变成 description 字段
    python_expression: str


class Node(msgspec.Struct):
    # 递归类型: 链表节点 (经 Optional 自引用, $ref 成环)
    val: int
    next: Optional['Node'] = None


class MutualA(msgspec.Struct):
    # 互递归类型: A.b -> B, B.a -> Optional[A] (字符串前向注解, 跨 $defs 条目互相引用成环)
    b: 'MutualB'


class MutualB(msgspec.Struct):
    a: Optional['MutualA'] = None


#: 非递归 schema 样本: 顶层 array + items.$ref (List[X] 参数形态)
_non_recursive_raw = msgspec.json.schema(List[ArgObj])

#: 递归 schema 样本: 顶层 $ref + $defs 内部再引用自身成环
_recursive_raw = msgspec.json.schema(Node)


def test_deref_expands_non_recursive():
    ''' 非递归 schema: $ref 全量展开, 结果无 $ref / 无 $defs (历史行为回归; 容器顶层形态的死 $defs 键
    按 "结果完全展开" 契约剥除 —— master 请求曾携带该死键, 字节级差异、语义等价) '''
    result = deref(_non_recursive_raw)

    assert result == {
        "type": "array",
        "items": {"title": "ArgObj", "type": "object", "properties": {"python_expression": {"type": "string"}}, "required": ["python_expression"]},
    }


def test_deref_expands_top_level_ref_to_definition_body():
    ''' 顶层 $ref (非递归 Struct 本体): deref 结果即定义体 (历史行为回归; 此形态经 $ref 分支展开,
    天然无死 $defs 键 —— 与 master 字节级一致, 死键差异仅存在于容器顶层形态, 见上) '''
    result = deref(msgspec.json.schema(ArgObj))

    assert result == {"title": "ArgObj", "type": "object", "properties": {"python_expression": {"type": "string"}}, "required": ["python_expression"]}
    assert "$ref" not in result and "$defs" not in result


def test_deref_rejects_recursive_with_guidance():
    ''' 递归 schema: deref 抛 RecursiveSchemaError (取代裸 RecursionError), 消息含能力指引关键词 '''
    with pytest.raises(RecursiveSchemaError, match = "recursive schema reference 'Node'"):
        deref(_recursive_raw)

    with pytest.raises(RecursiveSchemaError, match = "schema_refs"):
        deref(_recursive_raw)


def test_deref_rejects_mutually_recursive():
    ''' 互递归 A↔B (跨 $defs 条目成环): deref 抛 RecursiveSchemaError (回归锚 —— 未来重构若
    弱化 ref_stack 沿 properties/组合键的递归传递, 自引用用例可能仍过而互递归静默退化为裸 RecursionError) '''
    with pytest.raises(RecursiveSchemaError, match = "recursive schema reference"):
        deref(msgspec.json.schema(MutualA))

    with pytest.raises(RecursiveSchemaError, match = "schema_refs"):
        deref(msgspec.json.schema(MutualB))


def test_deref_allows_repeated_non_cyclic_refs():
    ''' 同一 $defs 条目在兄弟位置被多次引用: 非环 (栈式检测不误伤菱形引用) '''

    class Twin(msgspec.Struct):
        left: ArgObj
        right: ArgObj
    result = deref(msgspec.json.schema(Twin))
    assert result["properties"]["left"] == result["properties"]["right"]
    assert "$ref" not in json.dumps(result)


def test_resolve_schema_non_recursive_zero_drift():
    ''' 非递归 × {refs_allowed True/False}: 恒 deref, 两者结果一致 (全平台零漂移) '''
    resolved_true = resolve_schema(_non_recursive_raw, refs_allowed = True)
    resolved_false = resolve_schema(_non_recursive_raw, refs_allowed = False)

    assert resolved_true == deref(_non_recursive_raw)
    assert resolved_false == deref(_non_recursive_raw)


def test_resolve_schema_recursive_passthrough_when_allowed():
    ''' 递归 + refs_allowed: 透传 raw ($ref/$defs 保留); 返回独立深拷贝, 原地修改不回染 raw '''
    resolved = resolve_schema(_recursive_raw, refs_allowed = True)

    assert resolved == _recursive_raw
    assert resolved is not _recursive_raw

    resolved["$defs"]["Node"]["properties"]["val"]["type"] = "mutated"
    assert _recursive_raw["$defs"]["Node"]["properties"]["val"]["type"] == "integer"


def test_resolve_schema_recursive_rejected_when_not_allowed():
    ''' 递归 + 无能力位: fail-fast 抛 RecursiveSchemaError '''
    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        resolve_schema(_recursive_raw, refs_allowed = False)


def test_wrap_result_schema_recursive():
    ''' 递归 raw 包装: properties.result 为 {"$ref": ...}, $defs 提升到包装文档根 '''
    wrapped = wrap_result_schema(_recursive_raw)

    assert wrapped["type"] == "object"
    assert wrapped["properties"]["result"] == {"$ref": "#/$defs/Node"}
    assert wrapped["required"] == ["result"]
    assert wrapped["$defs"] == _recursive_raw["$defs"]


def test_wrap_result_schema_recursive_container_without_top_ref():
    ''' 递归容器 (List[Node]): 顶层无 $ref 但含内部 $ref + $defs —— inner 去除 $defs, $defs 提升到包装根 '''
    raw = msgspec.json.schema(List[Node])

    wrapped = wrap_result_schema(raw)

    assert wrapped["properties"]["result"] == {"type": "array", "items": {"$ref": "#/$defs/Node"}}
    assert "$defs" not in wrapped["properties"]["result"]
    assert wrapped["$defs"] == raw["$defs"]


def test_wrap_result_schema_non_recursive_zero_drift():
    ''' 非递归 (deref 结果) 包装: 与手工包装逐字节一致 (零漂移), 且无 $defs 键 '''
    deref_result = deref(_non_recursive_raw)

    wrapped = wrap_result_schema(deref_result)

    expected = {"type": "object", "properties": {"result": deref_result}, "required": ["result"]}
    assert wrapped == expected
    assert json.dumps(wrapped) == json.dumps(expected)  # 含键序的逐字节一致
    assert "$defs" not in wrapped


def test_wrap_result_schema_simple_type_no_defs_key():
    ''' 简单类型 schema ({"type":"integer"}): 包装无 $ref 无 $defs, 不产出空 $defs 键 '''
    wrapped = wrap_result_schema({"type": "integer"})

    assert wrapped == {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"]}


def test_shallow_expand_recursive():
    ''' 递归顶层 $ref 浅展开: 定义体提升为顶层 (字面 object), $defs 保留根部, 内部 $ref 原样保留 '''
    expanded = shallow_expand(_recursive_raw)

    assert expanded["type"] == "object"
    assert expanded["properties"]["val"] == {"type": "integer"}
    assert expanded["properties"]["next"]["anyOf"][1] == {"$ref": "#/$defs/Node"}
    assert expanded["$defs"] == _recursive_raw["$defs"]


def test_shallow_expand_non_ref_returns_asis():
    ''' 顶层无 $ref (deref 结果): 原样返回 (同一对象, 零开销零漂移) '''
    deref_result = deref(_non_recursive_raw)

    assert shallow_expand(deref_result) is deref_result
    simple_schema = {"type": "integer"}
    assert shallow_expand(simple_schema) is simple_schema
