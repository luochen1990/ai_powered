from pydantic.dataclasses import dataclass
from typing import Optional
from ai_powered import ai_powered

@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[str]

@ai_powered
def gen_user_info() -> UserInfo:
    ...

@ai_powered
def gen_self_introduction_for(user_info: UserInfo) -> str:
    ''' generate self introduction based on user info '''
    ...

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    ...

def test_gen_user_info():
    user_info = gen_user_info()
    print(f"{user_info =}")

def test_gen_self_introduction():
    user_info = gen_user_info()
    self_introduction = gen_self_introduction_for(user_info)
    print(f"{self_introduction =}")

def test_extract_identity():
    user_info = gen_user_info()
    self_introduction = gen_self_introduction_for(user_info)
    user_info_extracted = extract_user_info(self_introduction)
    assert user_info_extracted == user_info
