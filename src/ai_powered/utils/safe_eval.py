def safe_eval(expr: str):
    return eval(expr, {"__builtins__": None}, {})
