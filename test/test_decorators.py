from dataclasses import dataclass
from typing import Optional
from ai_powered import ai_powered

@ai_powered
def add(a: int, b: int) -> int:
    ...

@ai_powered
def gen_random_list() -> list[int]:
    ...

@ai_powered
def sort(xs: list[int]) -> list[int]:
    ''' sort elements in decreasing order '''
    ...

@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[str]

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    ...

class TestDecorator:

    def test_add(self):
        assert add(1, 1) == 2
        assert add(1, 2) == 3

    def test_sort(self):
        xs = [3, 1, 2]
        assert sort(xs) == sorted(xs, reverse=True)

    def test_sort_random(self):
        xs = gen_random_list()
        print(f"{xs =}")
        assert sort(xs) == sorted(xs, reverse=True)

    def test_extract_user_info(self):
        self_introduction = "Hello, my name is Alice, and I am from USA. I am 10 years old"
        user_info = extract_user_info(self_introduction)
        assert user_info == UserInfo(name="Alice", country="USA", age="10")
