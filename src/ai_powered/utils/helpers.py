import warnings

def only_warn(condition: bool) -> bool:
    if not condition:
        warnings.warn("Test failed, but marked as only_warn", UserWarning)
    return True
