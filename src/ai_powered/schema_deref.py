'''
职责: function calling schema 的 $ref 处理中枢 —— deref ($ref 展开) + 递归环检测 + 能力门控 schema 变体选择 + $defs 根提升.

历史背景: 2024 年因大部分 LLM 服务商不支持 schema 中的 $ref 而全量 deref, 但 deref 无法展开环状引用
(递归类型 → 裸 RecursionError). 该策略已被 capability-gated 策略取代 (issue 18):
  - deref: $ref 全量展开 (非递归 schema 的历史行为, 全平台零语义漂移 —— 容器顶层死 $defs 键按契约剥除, master 曾携带), 遇环抛 RecursiveSchemaError (fail-fast)
  - resolve_schema: 能力门控 SSOT —— 模型通道声明 schema_refs 能力时, 递归 schema 原样透传 ($ref/$defs 保留)
  - wrap_result_schema / shallow_expand: 透传 $ref 时的文档组装 —— $defs 必须留在文档根
    (JSON Schema fragment 从文档根解析, 内部 $ref 不提升根 $defs 即悬空)

msgspec 产出的 schema 形态 (含非递归 Struct): 顶层即 `{'$ref': '#/$defs/X', '$defs': {...}}`;
递归类型在 $defs 内部再引用自身成环.
'''
from copy import deepcopy
from typing import Any


class RecursiveSchemaError(ValueError):
    ''' 递归引用检测: deref 无法展开环状 $ref; 需要模型通道支持 schema 引用透传 (issue 18) '''


def deref(schema: dict[str, Any]) -> dict[str, Any]:
    '''
    将 msgspec 给出的 json schema 中的 $ref 字段替换为具体的 schema (非递归 schema 的历史行为).

    这一行为是符合标准的，见:  https://json-schema.org/understanding-json-schema/structuring#dollarref

    大部分 LLM 服务商不支持 $ref 字段，所以需要将其展开以获得更好的效果;
    环状引用 (递归类型) 无法展开, 抛 RecursiveSchemaError fail-fast (由 resolve_schema 按能力门控).

    契约: 结果完全展开 —— 无 $ref (全部内联) 且无 $defs (顶层 $defs 在全量展开后已无引用, 一并剥除).
    '''
    defs = schema.get("$defs", {})

    # JSON Schema 中可能出现子 schema (含 $ref) 的容器键, 按值的形态分两类:
    # - 单 schema 形态: 值本身就是一个 schema (dict 或 bool, 如 "items": false)
    #   - items: array 元素
    #   - additionalProperties: map 值
    # - dict[str, schema] 形态: 值是属性名到 schema 的映射
    #   - properties: 对象属性
    # - list[schema] 形态: 值是 schema 列表
    #   - prefixItems: 定长 tuple 元素
    #   - anyOf / oneOf / allOf: 组合
    # 只对这些键递归, 避免误处理 title/description/required 等非 schema 值.
    single_schema_keys = ("items", "additionalProperties")
    dict_schema_keys = ("properties",)
    list_schema_keys = ("prefixItems", "anyOf", "oneOf", "allOf")

    def _deref(schema: dict[str, Any], ref_stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        ''' 展开一层 schema; ref_stack 为当前展开路径上的 ref 名集合 (命中即环, 兄弟位置重复引用不误伤) '''
        if '$ref' in schema:
            ref_name = schema["$ref"].split('/')[-1]
            if ref_name in ref_stack:
                raise RecursiveSchemaError(
                    f"recursive schema reference '{ref_name}' detected: deref cannot expand cyclic $ref; "
                    f"this model channel does not support schema refs (see issue 18) — use a model with "
                    f"schema_refs capability or avoid recursive types"
                )
            return _deref(defs[ref_name], ref_stack | {ref_name})
        result = dict(schema)
        for key in single_schema_keys:
            if isinstance(result.get(key), dict):
                result[key] = _deref(result[key], ref_stack)
        for key in dict_schema_keys:
            value = result.get(key)
            if isinstance(value, dict):
                result[key] = {k: _deref(v, ref_stack) for k, v in value.items()}
        for key in list_schema_keys:
            value = result.get(key)
            if isinstance(value, list):
                result[key] = [_deref(v, ref_stack) if isinstance(v, dict) else v for v in value]
        return result

    # 全量展开后顶层 $defs 已无引用 (环状引用在上面 fail-fast), 剥除死键以满足 "结果完全展开" 契约
    resolved = _deref(schema)
    resolved.pop("$defs", None)
    return resolved


def resolve_schema(raw: dict[str, Any], *, refs_allowed: bool) -> dict[str, Any]:
    ''' 能力门控 SSOT: 按模型通道的 schema_refs 能力选择 schema 变体.

    非递归 → 恒 deref (全平台零语义漂移; 容器顶层 schema 的死 $defs 键按 deref "结果完全展开"契约剥除, master 请求会携带该死键);
    递归 → refs_allowed ? 透传 raw 的独立深拷贝 ($ref/$defs 保留) : 抛 RecursiveSchemaError (fail-fast)

    返回值契约 (M3 防御): 透传分支返回独立深拷贝, 与 raw 无任何共享子树, 调用方可安全原地修改
    (下游 wrap_result_schema / shallow_expand 会将子树挂入请求文档, 共享即契约脆弱);
    deref 分支为构造的新 dict 树 (未被容器键覆盖的嵌套值可能与输入共享, 输出仅进请求序列化, 勿原地修改).
    '''
    try:
        return deref(raw)
    except RecursiveSchemaError:
        if refs_allowed:
            return deepcopy(raw)
        raise


def wrap_result_schema(schema: dict[str, Any]) -> dict[str, Any]:
    ''' 把 resolved return schema 包装为 {"type":"object","properties":{"result":...},"required":["result"]} 请求文档.

    顶层含 $ref → inner = {"$ref": ...}, schema 顶层 $defs 提升到包装文档根
    (JSON Schema fragment 从文档根解析, 不提升则悬空);
    顶层无 $ref → inner = 去除 $defs 的原 schema (内部 $ref 仍由根 $defs 解析).
    无 $defs 则不产出空 $defs 键; 非递归输入 (deref 结果) 输出与历史手工包装逐字节一致 (零漂移).
    '''
    defs = schema.get("$defs")
    if '$ref' in schema:
        inner: dict[str, Any] = {"$ref": schema["$ref"]}
    else:
        inner = {k: v for k, v in schema.items() if k != "$defs"}
    wrapped = {"type": "object", "properties": {"result": inner}, "required": ["result"]}
    if defs:
        wrapped["$defs"] = defs
    return wrapped


def shallow_expand(schema: dict[str, Any]) -> dict[str, Any]:
    ''' 顶层 $ref 内联展开一层 (定义体提升为顶层), $defs 保留在结果根部; 顶层无 $ref 原样返回 (同一对象).

    用途: 工具参数 schema 根需字面 object (msgspec schema(Struct) 顶层恒为 $ref 形式),
    内部 $ref 由根部 $defs 继续解析 (递归透传时尤为关键).
    '''
    if '$ref' not in schema:
        return schema
    ref_name = schema["$ref"].split('/')[-1]
    defs = schema.get("$defs", {})
    expanded = dict(defs[ref_name])
    expanded["$defs"] = defs
    return expanded
