from ai_powered import ai_powered

@ai_powered
def binary_search(arr: list[int], x: int) -> int:
    ''' Returns the index of x in arr if present, else -1 '''
    ...

def test_binary_search():

    assert binary_search([1, 2, 100, 103, 104, 105], 103) == 3
    assert binary_search([1, 2, 100, 103, 104, 105], 2) == 1
