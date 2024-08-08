import pytest
from ai_powered import ai_powered

@ai_powered
async def add(a: int, b: int) -> int:
    ...

@pytest.mark.asyncio
async def test_add_async():

    assert (await add(1, 1)) == 2
    assert (await add(1, 2)) == 3
