from ai_powered import ai_powered

@ai_powered
def gen_random_list() -> list[int]:
    ...

@ai_powered
def sort(xs: list[int]) -> list[int]:
    ''' sort elements in decreasing order '''
    ...

def test_sort():
    xs = [3, 1, 2]
    assert sort(xs) == sorted(xs, reverse=True)

def test_sort_random():
    xs = gen_random_list()
    print(f"{xs =}")
    assert sort(xs) == sorted(xs, reverse=True)
