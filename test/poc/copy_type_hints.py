import inspect
from typing import Any, get_type_hints

# 假设这是你的函数 g，带有复杂的参数类型签名
def g(a: int, b: str, c: float) -> bool:
    # 函数体
    return True

# 获取函数 g 的类型提示
# 获取函数 g 的签名
sig_g = inspect.signature(g)


# 定义函数 f，复用 g 的参数类型定义
def f(*args: Any, **kwargs: Any):
    # 调用 g 并进行后处理
    result = g(*args, **kwargs)
    # 后处理逻辑
    processed_result = not result  # 示例后处理逻辑
    return processed_result

# 使用函数注解来复用 g 的参数类型定义
f.__annotations__ = {
    name: param.annotation
    for name, param in sig_g.parameters.items()
}
f.__annotations__['return'] = sig_g.return_annotation  # 设置返回类型

# 现在函数 f 的参数类型定义与 g 相同
print(get_type_hints(f))

f()
