
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
    if '$ref' and '$defs' in schema:
        ref_name = schema["$ref"].split('/')[-1]
        return schema["$defs"][ref_name]
    else:
        return schema
