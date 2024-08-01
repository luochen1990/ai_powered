from dataclasses import dataclass
from typing_extensions import Optional
from ai_powered import ai_powered

@ai_powered
def is_same_country(a: str, b: str) -> bool:
    ''' judge if a and b is probably alias of the same country '''
    ...

@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[int]

    def consistent_with(self, other: 'UserInfo') -> bool:
        name_check = self.name.lower() == other.name.lower()
        country_check = is_same_country(self.country, other.country) if (self.country is not None and other.country is not None) else self.country == other.country
        age_check = self.age == other.age
        return name_check and country_check and age_check

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    '''
    Obtain complete information as much as possible, unless it is truly not mentioned;
    when identifying countries, match them to the closest real existing country
    '''
    ...

def user_info_match(a: UserInfo, b: UserInfo) -> bool:
    ''' judge if the two user info is consistent '''
    ...

def test_extract_user_info_1():
    self_introduction = "Hello, my name is Alice, and I am from USA. I am 10 years old"
    user_info = extract_user_info(self_introduction)
    print(f"{user_info =}")

    assert user_info.consistent_with(UserInfo("Alice", "USA", 10))

def test_extract_user_info_2():
    self_introduction = "Hello, Iama ten years old amecan boy namd bob"
    user_info = extract_user_info(self_introduction)
    print(f"{user_info =}")

    assert user_info.consistent_with(UserInfo("Bob", "USA", 10))
