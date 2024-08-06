
from typing import Any

def deref(schema: dict[str, Any]) -> dict[str, Any]:
    '''
    msgspec 给出的 json schema 会包含 $ref 字段，这个函数用于将 $ref 字段替换为具体的 schema

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
