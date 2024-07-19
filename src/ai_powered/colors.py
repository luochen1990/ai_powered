def red(s: str) -> str:
    return '\033[91m' + s + '\033[0m'

def orange(s: str) -> str:
    return '\033[33m' + s + '\033[0m'

def yellow(s: str) -> str:
    return '\033[93m' + s + '\033[0m'

def green(s: str) -> str:
    return '\033[32m' + s + '\033[0m'

def gray(s: str) -> str:
    return '\033[90m' + s + '\033[0m'
