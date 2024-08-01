from typing_extensions import Any
from ai_powered import ai_powered

@ai_powered
def gen_json_data(depth: int) -> Any:
    ''' random generate some valid json data for test usage '''
    ...

def test_gen_json_data():
    some_json_data = gen_json_data(depth=5)
    print(f"{some_json_data =}")
