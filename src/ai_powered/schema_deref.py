
from typing import Any

def deref(schema: dict[str, Any]) -> dict[str, Any]:
    '''
    msgspec 给出的 json schema 会包含 $ref 字段，这个函数用于将 $ref 字段替换为具体的 schema

    这一行为是符合标准的，见:  https://json-schema.org/understanding-json-schema/structuring#dollarref

    但 OpenAI 及大部分 LLM 服务商尚不支持 $ref 字段，所以需要将其展开以获得更好的效果.

    当然这一做法也有其局限性，比如无法处理环状引用，但这只能等到服务商支持 $ref 字段后再解决.

    {'$ref': '#/$defs/ArgObj', '$defs': {'ArgObj': {'title': 'ArgObj', 'type': 'object', 'properties': {'python_expression': {'type': 'string'}}, 'required': ['python_expression']}}}

    最终返回结果的最外层形如:

    return {
        "type": "object",
        "properties": properties,
        "required": [param.name for param in sig.parameters.values() if param.default == inspect.Parameter.empty],
    }
    '''

    #TODO: 环状引用检测（递归类型，如 Nat，List）

    defs = schema.get("$defs", {})

    def _deref(schema: dict[str, Any]) -> dict[str, Any]:
        if '$ref' in schema:
            ref_name = schema["$ref"].split('/')[-1]
            return _deref(defs[ref_name])
        elif 'properties' in schema:
            properties = {key: _deref(value) for key, value in schema['properties'].items()}
            return schema | {"properties": properties}
        else:
            return schema

    return _deref(schema)
