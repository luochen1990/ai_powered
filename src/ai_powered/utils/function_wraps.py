from typing import Any, Callable, TypeVar, cast
from typing_extensions import ParamSpec, Concatenate

P = ParamSpec('P')
R = TypeVar('R')
S = TypeVar('S')


def wraps(origin_fn: Callable[P, R]) -> Callable[[Callable[..., Any]], Callable[P, R]]:

    def wrapper(fn: Callable[..., Any]) -> Callable[P, R]:
        return fn  #type: ignore

    return wrapper


def wraps_method_arguments_type(origin_fn: Callable[P, Any]) -> Callable[[Callable[Concatenate[S, ...], R]], Callable[Concatenate[S, P], R]]:

    def wrapper(fn: Callable[..., R]) -> Callable[Concatenate[S, P], R]:

        @wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return fn(*args, **kwargs)

        return cast(Callable[Concatenate[S, P], R], inner)

    return wrapper


class Add:

    def add(self, x: int, y: int) -> int:
        return x + y


class A:

    @wraps_method_arguments_type(Add().add)
    def add(self, *args: list[Any], **kwargs: dict[str, Any]) -> int:
        return 1


a = A()
r = a.add(1, 2)
