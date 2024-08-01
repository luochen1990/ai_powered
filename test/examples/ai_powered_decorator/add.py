from ai_powered import ai_powered

@ai_powered
def add(a: int, b: int) -> int:
    ...

def test_add():

    assert add(1, 1) == 2
    assert add(1, 2) == 3
