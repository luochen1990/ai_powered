from dataclasses import dataclass
from typing import Optional
from ai_powered import ai_powered

@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[str]

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    ...

def test_extract_user_info():
    self_introduction = "Hello, my name is Alice, and I am from USA. I am 10 years old"
    user_info = extract_user_info(self_introduction)
    assert user_info == UserInfo(name="Alice", country="USA", age="10")
