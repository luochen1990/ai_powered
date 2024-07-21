from ai_powered import ai_powered

@ai_powered
def python_expression_generator(expr: str) -> str:
    ''' 将用户输入的数学表达式转换成合法的python表达式, 注意不要使用任何未定义的函数，如果用户表达式中有类似函数调用的表达，请转换为python内置函数或语法 '''
    ...

def calculator(expr: str) -> int:
    ''' calculate the result of the math expression '''
    py_expr = python_expression_generator(expr)
    print(f"{py_expr =}")
    x = eval(py_expr)
    return int(x)

def test_calculator():
    assert calculator("1+1") == 2
    assert calculator("1+2*3") == 7
    assert calculator("2^10+3*4") == 1036
    assert calculator("2^10+sum(1,10)") == 1079
