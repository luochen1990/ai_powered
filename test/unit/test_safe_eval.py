'''
职责: safe_eval 的离线单元测试 —— 攻击样本表回归 + 良性语料差分对拍 (issue #10 DoD).

覆盖:
  - 攻击样本表: import/dunder 爬链/getattr 逃逸/exec-eval 套娃/f-string/format-string 注入/DoS 守卫
    全部被拒 (拒绝 = 抛 SafeEvalViolation, 非静默的错误结果); 语句语法按构造不可解析 (SyntaxError)
  - 良性语料差分对拍: 门内结果 == 无沙箱 eval 基准
  - 语义保持: 求值返回原生值; SyntaxError 与运行期错误自然传播 (与 eval 同形)
'''
import pytest

from ai_powered.utils.safe_eval import SafeEvalViolation, safe_eval

#: 攻击样本表 (issue DoD 全项): 拒绝 = pytest.raises(SafeEvalViolation)
ATTACK_SAMPLES = {
    "import dunder": "__import__('os').system('id')",
    "exec nesting": 'exec("import os")',
    "eval nesting": "eval('__import__(\"os\")')",
    "class chain": "().__class__",
    "subclass crawl": "().__class__.__base__.__subclasses__()",
    "getattr escape": "getattr((), '__class__')",
    "getattr chain": "getattr(getattr((), '__class__'), '__base__')",
    "lambda escape": "(lambda: __import__('os'))()",
    "f-string injection": "f'{().__class__}'",
    "format-string attribute crawl": '"{0.__class__}".format(())',
    "format_map crawl": '"{0.__class__}".format_map({0: ()})',
    "comprehension crawl": "[c for c in ().__class__.__base__.__subclasses__()]",
    "name injection": "str(__builtins__)",
    "globals leak": "str(globals())",
    "locals leak": "str(locals())",
    "open file": "open('/etc/passwd')",
    "pow bomb nested": "10 ** 10 ** 10",
    "pow bomb literal": "2 ** 999999999",
    "pow bomb folded": "9 ** (9 ** 9)",
    "repetition bomb": "'a' * 999999999",
    "repetition bomb reversed": "999999999 * 'a'",
    "repetition bomb folded": "'a' * (10 ** 9)",
    "list repetition bomb folded": "[0] * (2 ** 30)",
    "repetition bomb shift": "[0] * (1 << 30)",
    "repetition bomb bitwise or": "'a' * (0 | 1000000000)",
    "repetition bomb huge fold": "[0] * ((10 ** 7) ** 1000)",
    "repetition bomb invert": "[0] * ~(-10 ** 9)",
    "repetition bomb boolop": "[0] * (10 ** 9 or 1)",
    "repetition bomb ifexp": "[0] * (10 ** 9 if 1 else 0)",
    "repetition bomb subscript": "[0] * [10 ** 9][0]",
    "repetition bomb chained mult": "(([0] * 1000) * 1000) * 1000",
    "repetition bomb concat composed": "([0] * 10 ** 5 + [0] * 10 ** 5) * 10 ** 5",
    "repetition bomb slice composed": "(([0] * 10 ** 5)[0:10 ** 5]) * 10 ** 5",
    "repetition bomb ifexp composed": "(([0] * 10 ** 5) if 1 else ()) * 10 ** 5",
    "repetition bomb subscript composed": "(([0] * 10 ** 5,)[0]) * 10 ** 5",
    "repetition bomb starred composed": "[0, *([0] * 10 ** 5)] * 10 ** 5",
    "concat doubling bomb": "([0] * 10 ** 8 + [0] * 10 ** 8)",
    "repetition inf count": "'a' * (1e308 * 10)",
    "format width bomb": "'%200000000d' % 1",
    "format arg width bomb": "'%*d' % (200000000, 1)",
    "format precision bomb": "'%.99999999f' % 1",
    "format composed bomb": "'%s' % ([0] * 10 ** 8)",
    "format bytes width bomb": "b'%999999999d' % 1",
    "format concat template bomb": "('%20' + '0000000d') % 1",
    "pow result size bomb": "(1 << 39999) ** 4096",
    "pow result size chained bomb": "((1 << 39999) ** 4096) * ((1 << 39999) ** 4096)",
    "pow ifexp base bomb": "((1 << 39999) if 1 else 0) ** 4096",
    "pow boolop base bomb": "((1 << 39999) or 0) ** 4096",
    "pow subscript base bomb": "((10 ** 4096,)[0]) ** 4096",
    "shift bomb": "1 << (10 ** 9)",
    "shift huge base": "((1 << 12000) ** 3) << 1",
    "shift composed pow bomb": "((1 << (10 ** 9)) if 1 else 0) ** 4096",
    "annihilator ifexp pow smuggle": "(1 if ((1 << 12000) ** 3) * 0 else (1 << 39999)) ** 4096",
    "annihilator ifexp shift smuggle": "(1 if ((1 << 12000) ** 3) * 0 else (1 << 39999)) << 10",
    "annihilator ifexp count smuggle": "[0] * (1 if ((1 << 12000) ** 3) * 0 else 10 ** 9)",
    "annihilator boolop pow smuggle": "((((1 << 12000) ** 3) * 0) or (1 << 39999)) ** 4096",
    "annihilator boolop shift smuggle": "((((1 << 12000) ** 3) * 0) or (1 << 39999)) << 10",
    "boolop huge value count smuggle": "[0] * (((1 << 12000) ** 3) or 1)",
    "pow huge base nested": "((2 ** 4096) ** 4096) ** 4096",
    "fold digit-limit escape via mult": "[0] * (((10 ** 7) ** 1000) ** 1)",
    "fold digit-limit escape via pow": "2 ** (((10 ** 7) ** 1000) ** 1)",
    "walrus assignment": "(x := 5)",
    "dunder argument name": "(lambda __x: __x)",
    "dunder comprehension var": "[__x for __x in ()]",
    "method not whitelisted": "'a'.__class__",
    "fromkeys not whitelisted": "dict.fromkeys([1, 2])",
    "compile not callable": "compile('1', '<s>', 'eval')",
    "lambda direct call": "(lambda: 1)()",
    "code too long": "0 + " * 3000 + "0",
}

#: 语句语法样本: 纯表达式求值按构造不可解析 (SyntaxError), 即 "全拒" 的另一面
STATEMENT_SAMPLES = {
    "import statement": "import os",
    "from import": "from os import system",
    "assignment": "x = 1",
    "del": "x = 1; del x",
    "global": "global x",
    "with": "with open('x') as f: pass",
    "class def": "class A: pass",
    "def": "def f(): pass",
    "async def": "async def f(): pass",
    "raise": "raise Exception('x')",
}


@pytest.mark.parametrize("sample", ATTACK_SAMPLES.values(), ids = ATTACK_SAMPLES.keys())
def test_attack_samples_rejected(sample: str):
    ''' 攻击样本全部被拒: SafeEvalViolation (清晰异常), 而非静默的错误结果 '''
    with pytest.raises(SafeEvalViolation):
        safe_eval(sample)


@pytest.mark.parametrize("sample", STATEMENT_SAMPLES.values(), ids = STATEMENT_SAMPLES.keys())
def test_statement_syntax_rejected(sample: str):
    ''' 语句不在 (求值) 契约内: SyntaxError 自然传播, 与 eval 同形 '''
    with pytest.raises(SyntaxError):
        safe_eval(sample)


#: 文档化的保守拒绝 (fail-closed 的良性附带面): 裸 eval 可算, 但静态不可判界而拒绝
CONSERVATIVE_REJECTIONS = {
    "literal % formatting": "'%d items' % 3",  # 宽度/精度不可静态判界, 与 str.format 同类
    "ifexp mixed branch repetition": "([0] * 10 if 1 else 0) * 2",  # 分支类型混合 → 序列长度不可判
    "negative index composed repetition": "(('a' * 10 ** 8,)[-1]) * 10",  # 负下标不静态解析
    "dict repetition": "{1: 2} * 2",  # 非序列容器 × 数 → 长度不可判
    "unfoldable compare shift": "1 << (1 < 2)",  # Compare 结果不作折叠移位量
    "unfoldable compare pow base": "(1 < 2) ** 2",  # Compare 结果不作折叠底数
}


@pytest.mark.parametrize("sample", CONSERVATIVE_REJECTIONS.values(), ids = CONSERVATIVE_REJECTIONS.keys())
def test_documented_conservative_rejections(sample: str):
    ''' 保守拒绝类: 输入良性但静态不可判界 → SafeEvalViolation (回喂后可改写, 如 f-string/显式拼接) '''
    with pytest.raises(SafeEvalViolation):
        safe_eval(sample)


#: 良性语料差分对拍基准 = 无沙箱 eval (门内结果须与其一致)
BENIGN_CORPUS = [
    "1 + 2 * 3",
    "2 ** 10 + 3 ** 4",
    "9 ** 0.5",
    "2 ** -10",
    "2 ** 3 ** 2",
    "(1 << 100) ** 300",
    "(1 << 19999) ** 2",
    "1 << 10",
    "((1 << 12000) ** 3) * 0",
    "(1 if 1 else 2) + 3",
    "2 ** True",
    "2 ** (7 % 3)",
    "(1 << 10) + 5",
    "12 % 5",
    "5 | 3",
    "(-8) ** 0.5",
    "1400000000 * 2",
    "999999999 * 0",
    "[0] * 5 * 3",
    "[1, 2] * 3 * 2",
    "'ab' * 2",
    "('ab' + 'cd') * 10",
    "('a' + 'b') * 5",
    "True * 'a'",
    "[*{1, 2}]",
    "[0] * (2 or 3)",  # BoolOp 折叠后为良性 (计数 2)
    "(-7) // 2",
    "'ab' * 3",
    "(255).bit_length()",
    "(3.5).is_integer()",
    "(7).real",
    "abs(-5) + round(3.14159, 2)",
    "max(len('abc'), 4)",
    "sorted([3, 1, 2], reverse = True)",
    "'a,b,c'.split(',')",
    "'-'.join(str(x) for x in range(5))",
    "sum(x * x for x in range(5))",
    "[x.upper() for x in ['a', 'b']]",
    "{'k': 1, 'j': 2}",
    "list(zip([1, 2], 'ab'))",
    "list(reversed([1, 2, 3]))",
    "list(map(lambda x: x + 1, [1, 2, 3]))",
    "f'{2 + 3} ok'",
    "divmod(17, 5)",
    "'abcdef'[1:4]",
    "(1, 2, 3)[::-1]",
    "True and not False",
    "'a' if 1 < 2 else 'b'",
    "format(3.14159, '.2f')",
    "set([1, 2, 2, 3])",
    "int('42') + float('0.5')",
    "max(*[1, 5, 3])",
]


@pytest.mark.parametrize("code", BENIGN_CORPUS)
def test_benign_differential(code: str):
    ''' 门内结果 == 无沙箱 eval 基准 (差分对拍) '''
    assert safe_eval(code) == eval(code)  # noqa: S307 对拍基准: 无沙箱 eval 本身就是被测对象


def test_calculator_scenario():
    ''' 活链路回归: use_calculator 的典型输入 (数学幂运算 + 内置函数) '''
    assert safe_eval("2 ** 10 + 3 ** 4") == 1105
    assert safe_eval("round(10 / 3, 2)") == 3.33


def test_return_value_not_stringified():
    ''' 语义保持: 返回原生值 (非字符串化) '''
    assert safe_eval("[1, 2]") == [1, 2]
    assert isinstance(safe_eval("sorted({3, 1})"), list)


@pytest.mark.parametrize("code,exc", [
    ("1 / 0", ZeroDivisionError),
    ("undefined_name", NameError),
    ("[1][5]", IndexError),
    ("1 +", SyntaxError),
])
def test_runtime_errors_propagate(code: str, exc: type[Exception]):
    ''' 运行期错误/语法错误自然传播 (与 eval 同形), 不伪装成安全拒绝 '''
    with pytest.raises(exc):
        safe_eval(code)


def test_violation_is_value_error():
    ''' 拒绝类型是 ValueError 子类: 调用方 (calculator 工具) 可按 ValueError 统一捕获 '''
    assert issubclass(SafeEvalViolation, ValueError)


def test_deep_nesting_no_uncontrolled_crash():
    ''' 深嵌套输入受控拒绝. 跨版本契约: 3.10/3.11 parse 期 RecursionError/MemoryError;
    3.12+ SyntaxError; parse 成功的版本则由节点数守卫拒绝 —— 无未捕获的失控路径 '''
    code = "-" * 5000 + "1"
    with pytest.raises((SafeEvalViolation, SyntaxError, RecursionError, MemoryError)):
        safe_eval(code)


def test_fold_memoization_no_cubic_amplification():
    ''' 守卫折叠带节点级 memo: 左结合长链的重复折叠不构成 O(n³) 放大 (无 memo 时 300 项链需分钟级) '''
    import time
    expr = "2 ** 4096 * " * 300 + "2"
    start = time.monotonic()
    safe_eval(expr)
    assert time.monotonic() - start < 20


def test_deep_mult_chain_rejected_gracefully():
    ''' 超深 Mult 链 (折叠递归超栈) → SafeEvalViolation (fail-closed), 而非未捕获 RecursionError '''
    with pytest.raises(SafeEvalViolation):
        safe_eval("1 * " * 2400 + "1")


def test_deep_pow_chain_rejected_gracefully():
    ''' 超深 Pow 链 (指数折叠递归超栈) → SafeEvalViolation (fail-closed), 而非未捕获 RecursionError '''
    with pytest.raises(SafeEvalViolation):
        safe_eval("2 ** " * 1500 + "2")
