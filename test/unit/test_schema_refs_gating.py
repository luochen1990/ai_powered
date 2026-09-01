'''
职责: schema_refs 能力门控的离线单元测试 —— 适配器构造期的 schema 变体选择 (issue 18 阶段 1)
+ MakeTool.schema 门控 + 平台能力注册表回归 (P2/C).

核心不变式:
  - 有 schema_refs 能力 + 递归 return schema → 请求参数透传 $ref, $defs 在文档根
  - 无 schema_refs + 递归 → 构造期 (装饰期) fail-fast 抛 RecursiveSchemaError
  - 非递归 + 有/无能力位 → 与历史行为零漂移 (无 $ref / 无 $defs)

全程离线: 适配器构造不触网, connection 传 cast 的 None (构造期只组装请求参数, 不发请求).
'''
import json
from typing import Any, cast

import msgspec
import pytest

from ai_powered.llm.adapter_selector import FunctionSimulatorSelector
from ai_powered.llm.adapters.chat_adapter import ChatFunctionSimulator
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.llm.adapters.structured_output_adapter import StructuredOutputFunctionSimulator
from ai_powered.llm.adapters.tools_adapter import ToolsFunctionSimulator
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import FALLBACK_FEATURES, KNOWN_PLATFORMS
from ai_powered.schema_deref import RecursiveSchemaError, deref
from ai_powered.tool_call import MakeTool, return_result_tool_param


class TreeNode(msgspec.Struct):
    ''' 递归类型样本: 树节点 (经 list 自引用成环) '''
    name: str
    children: list['TreeNode'] = []


class Point(msgspec.Struct):
    # 非递归类型样本 (msgspec 顶层同为引用形态, resolve 后展开); 不写 docstring 以免 description 字段干扰断言
    x: int
    y: int


#: 递归 raw return schema (msgspec 原始输出, FunctionSpec.return_schema 现在存这个形态)
_recursive_raw = msgspec.json.schema(TreeNode)

#: 非递归 raw return schema
_non_recursive_raw = msgspec.json.schema(Point)


def make_spec(return_schema: dict[str, Any]) -> FunctionSpec:
    ''' 含指定 return schema 的最小 FunctionSpec (对齐 conftest.make_spec 模式) '''
    return FunctionSpec("fn", "def fn(x: int) -> TreeNode", "doc", {"x": {"type": "integer"}}, return_schema)


def make_tools_simulator(spec: FunctionSpec, features: set[ModelFeature]) -> ToolsFunctionSimulator:
    return ToolsFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", features, {})


def response_schema_of(sim: GenericFunctionSimulator) -> dict[str, Any]:
    ''' 适配器的 response_format → json_schema.schema (递归透传断言的公共取值路径) '''
    response_format = cast(dict[str, Any], sim._param_response_format)
    return response_format["json_schema"]["schema"]


def tool_parameters_at(sim: GenericFunctionSimulator, index: int = 0) -> dict[str, Any]:
    ''' 适配器工具列表第 index 项的 function.parameters (tools 路径恒为 list) '''
    tools = cast(list[dict[str, Any]], sim._param_tools)
    return tools[index]["function"]["parameters"]


def assert_fully_expanded(payload: object) -> None:
    ''' 完全展开不变式: 序列化文本中不含任何 $ref / $defs '''
    serialized = json.dumps(payload)
    assert '$ref' not in serialized
    assert '$defs' not in serialized


def test_tools_adapter_recursive_passthrough_with_capability():
    ''' tools 适配器 + schema_refs + 递归: return_result 参数含根级 $defs, properties.result 为 $ref '''
    sim = make_tools_simulator(make_spec(dict(_recursive_raw)), {ModelFeature.tools, ModelFeature.schema_refs})

    params = tool_parameters_at(sim)
    assert params["type"] == "object"
    assert params["properties"]["result"] == {"$ref": "#/$defs/TreeNode"}
    assert params["$defs"] == _recursive_raw["$defs"]


def test_tools_adapter_recursive_rejected_without_capability():
    ''' tools 适配器 + 无 schema_refs + 递归: 构造期 (适配器 __post_init__) fail-fast '''
    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        make_tools_simulator(make_spec(dict(_recursive_raw)), {ModelFeature.tools})


def test_tools_adapter_non_recursive_zero_drift():
    ''' 非递归 + 有/无 schema_refs: 参数形态与历史一致 (无 $ref 无 $defs, 零漂移) '''
    historical_params = cast(dict[str, Any], return_result_tool_param(deref(_non_recursive_raw)))["function"]["parameters"]

    for features in [{ModelFeature.tools}, {ModelFeature.tools, ModelFeature.schema_refs}]:
        sim = make_tools_simulator(make_spec(_non_recursive_raw), features)
        assert tool_parameters_at(sim) == historical_params
        assert_fully_expanded(sim._param_tools)


def test_structured_adapter_recursive_passthrough_with_capability():
    ''' structured 适配器 + schema_refs + 递归: response_format 的 schema 含根级 $defs, properties.result 为 $ref '''
    spec = make_spec(dict(_recursive_raw))
    sim = StructuredOutputFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.structured_outputs, ModelFeature.schema_refs}, {})

    schema = response_schema_of(sim)
    assert schema["properties"]["result"] == {"$ref": "#/$defs/TreeNode"}
    assert schema["$defs"] == _recursive_raw["$defs"]


def test_structured_adapter_recursive_rejected_without_capability():
    ''' structured 适配器 + 无 schema_refs + 递归: 构造期 fail-fast '''
    spec = make_spec(dict(_recursive_raw))
    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        StructuredOutputFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.structured_outputs}, {})


def test_structured_adapter_non_recursive_zero_drift():
    ''' 非递归 + 无 schema_refs: response_format schema 与历史手工包装一致 (零漂移) '''
    spec = make_spec(_non_recursive_raw)
    sim = StructuredOutputFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.structured_outputs}, {})

    schema = response_schema_of(sim)
    assert schema == {"type": "object", "properties": {"result": deref(_non_recursive_raw)}, "required": ["result"]}
    assert_fully_expanded(schema)


def test_chat_adapter_recursive_rejected_without_capability():
    ''' chat 适配器 + 无 schema_refs + 递归: system_prompt 构造路径 fail-fast (chat 不特殊化, P4) '''
    spec = make_spec(dict(_recursive_raw))
    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        ChatFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", set(), {})


def test_chat_adapter_recursive_transparent_with_capability():
    ''' chat 适配器 + schema_refs + 递归: system prompt 中透传 raw schema ($ref 可见) '''
    spec = make_spec(dict(_recursive_raw))
    sim = ChatFunctionSimulator(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.schema_refs}, {})

    assert '#/$defs/TreeNode' in sim.system_prompt


def _make_tool_with_recursive_param() -> MakeTool:
    def count_nodes(tree: TreeNode) -> int:
        ''' count nodes of the tree '''
        return 1 + sum(count_nodes(c) for c in tree.children)

    return MakeTool(count_nodes)


def test_make_tool_schema_recursive_allowed():
    ''' MakeTool.schema(refs_allowed=True) + 递归参数: 顶层字面 object + 根 $defs, 内部 $ref 保留 '''
    param = _make_tool_with_recursive_param().schema(refs_allowed = True)

    parameters = cast(dict[str, Any], param)["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["tree"] == {"$ref": "#/$defs/TreeNode"}
    assert "TreeNode" in parameters["$defs"]


def test_make_tool_schema_recursive_rejected_by_default():
    ''' MakeTool.schema 默认 refs_allowed=False + 递归参数: fail-fast (历史调用点零漂移) '''
    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        _make_tool_with_recursive_param().schema()


def test_make_tool_schema_non_recursive_zero_drift():
    ''' 非递归参数: refs_allowed 有/无结果一致, 且与 deref 现状一致 (无 $ref 无 $defs) '''

    def add(a: int, b: int) -> int:
        ''' add two numbers '''
        return a + b

    tool = MakeTool(add)
    default_schema = tool.schema()
    allowed_schema = tool.schema(refs_allowed = True)

    assert default_schema == allowed_schema
    assert_fully_expanded(default_schema)


def test_fallback_features_exclude_schema_refs():
    ''' P2 回归: 开放平台兜底能力集 = 全能力减 schema_refs '''
    assert FALLBACK_FEATURES == set(ModelFeature) - {ModelFeature.schema_refs}


def test_known_models_schema_refs_distribution():
    ''' C 修正回归: 仅 openai 条目含 schema_refs, 其余平台 (含未来新增) 一律不含 —— 全平台遍历,
    新增平台自动纳入守护, 免手维护名单结构性过期 (zhipu 条目即漏网后补的实证) '''
    platforms = {p.platform_name: p for p in KNOWN_PLATFORMS}

    for model in platforms["openai"].known_model_list:
        assert ModelFeature.schema_refs in model.supported_features, model.model_name

    for platform in KNOWN_PLATFORMS:
        if platform.platform_name == "openai":
            continue
        for model in platform.known_model_list:
            assert ModelFeature.schema_refs not in model.supported_features, f"{platform.platform_name}/{model.model_name}"


def test_selector_end_to_end_recursive_passthrough():
    ''' 端到端: 能力选择 (structured_outputs 优先) + 递归透传 —— selector 构造一次即完成门控链路 '''
    spec = make_spec(dict(_recursive_raw))
    selector = FunctionSimulatorSelector(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.structured_outputs, ModelFeature.tools, ModelFeature.schema_refs}, {})

    assert isinstance(selector._selected_impl, StructuredOutputFunctionSimulator)
    schema = response_schema_of(selector._selected_impl)
    assert schema["properties"]["result"] == {"$ref": "#/$defs/TreeNode"}
    assert schema["$defs"] == _recursive_raw["$defs"]

    with pytest.raises(RecursiveSchemaError, match = "recursive"):
        FunctionSimulatorSelector(spec, cast(LlmConnection, None), "fake-model", {ModelFeature.tools}, {})
