from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec


P = ParamSpec('P')
R = TypeVar('R')


def wraps(origin_fn: Callable[P, R]) -> Callable[ [Callable[..., Any]], Callable[P, R]]:
    def wrapper(fn: Callable[..., Any]) -> Callable[P, R]:
        return fn #type: ignore
    return wrapper


def wraps_arguments_type(origin_fn: Callable[P, Any]) -> Callable[ [Callable[..., R]], Callable[P, R]]:
    def wrapper(fn: Callable[..., R]) -> Callable[P, R]:
        return fn #type: ignore
    return wrapper
