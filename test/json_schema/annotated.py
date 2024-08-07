import json
import msgspec
from typing_extensions import Annotated
from pydantic.dataclasses import dataclass
from typing import Optional

from ai_powered.schema_deref import deref

@dataclass
class UserInfo:
    name: str
    country: Annotated[Optional[str], "None is used for unknown cases"]
    age: Annotated[Optional[str], "None is used for unknown cases"]

if __name__ == "__main__":
    schema = msgspec.json.schema(UserInfo)
    print("schema =", json.dumps(schema, indent=2))
    print("deref(schema) =", json.dumps(deref(schema), indent=2))
