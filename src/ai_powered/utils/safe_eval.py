'''
职责: LLM 生成表达式的受控求值 (活链路: test/examples/chat_bot/use_calculator.py, doc/en/others.md).

修复 issue #10 属性链逃逸洞: 空 builtins 挡不住 ().__class__.__base__.__subclasses__() 式爬链
—— 旧实现 eval(expr, {"__builtins__": None}, {}) 的存量洞 (自 issue #7 B4 剥离).

设计 (自 commit 0f83781 SafeEvalExecutor 收编的最小纯函数修):
  - 安全 = AST 允许清单门 与 受控命名空间 的合取: 门堵通路 (fail-closed: 未知节点/dunder 标识符/
    白名单外调用与属性访问, 拒绝 = 抛 SafeEvalViolation); 命名空间让危险名字 (open/os/__import__)
    按构造不存在 (__builtins__ = 纯函数 dict 而非 None)
  - 刻意禁 str.format/format_map: 模板 "{0.__class__}" 的属性爬发生在运行期, AST 门看不见;
    f-string 是被 AST 门控的安全等价物
  - pow() 与 ** 算子同水位防护, 双防线 (llm-bench#12): 静态门 (_guard_pow_call: 仅接受 2/3 个
    字面位置参数, 星号/关键字静态不可判界 → 拒; 3 参 modpow 结果本受 mod 界定, 统一按前两参
    守卫, fail-closed 一致性) + 命名空间内 pow 恒为值级运行时守卫包装 (_guarded_pow, 按实参
    判界 —— 别名/高阶通路走私的 pow 值终将实际调用它, 在此兜住)
  - 语义保持: 纯表达式求值返回值; SyntaxError 与运行期错误 (除零/未定义名) 自然传播, 与 eval 同形

已知边界 (诚实标注):
  - 无 CPU/内存硬限制, DoS 仅有静态缓解: 幂/移位的底数、指数/移位量与结果位数, 以及序列尺寸
    (重复/拼接/切片构词, 含链式复合) 经有界常量折叠与静态长度上界判界; 纯数值乘积链无逐节点预算,
    以码长为硬上界 (~5MB 量级). 数值字面量及其算术/位运算/字面分支与下标组合可判, 字面纯而不可判时
    fail-closed 拒绝; 名字/调用/属性/f-string 宽度与 % 格式化 (与 str.format 同类被拒) 等动态量
    不在静态守卫覆盖内, bwrap 档兜底; 超深链 (数百项以上) 的静态分析递归深度受限而拒绝
  - Call 门只放行 白名单内置 ∪ 推导式绑定名: 直接调用 lambda 字面量 / 形参回调等高阶写法被拒
    (fail-closed 语义, 异常回喂调用方可改写); 经推导式绑定的名字可被调用, 其值 (含 lambda) 仍由门内构造
  - 幂指数须静态可折叠: 含变量指数 (如 [2 ** k for k in range(64)] 的 k) 被拒
'''
import ast
import builtins
import operator
from collections.abc import Sequence
from typing import Any, Callable


class SafeEvalViolation(ValueError):
    ''' 表达式触犯允许清单 (安全边界事件): 专用类型使拒绝语义可测试, 调用方可统一按 ValueError 捕获 '''


#: 受控命名空间: 纯函数 + 数据构造子 (不含 print/异常类型 —— 纯表达式求值用不到;
#: 不含 open/getattr/globals 等一切带副作用或反射能力的名字; pow 为纯数学函数, 双防线防护见 _guarded_pow)
_PURE_BUILTINS_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "range", "repr", "reversed", "round",
    "set", "sorted", "str", "sum", "tuple", "zip",
)


def _guarded_pow(base: Any, exp: Any, mod: Any = None) -> Any:
    ''' pow 的值级运行时守卫 (llm-bench#12): 静态门守 "名字叫 pow 的调用", 别名/高阶通路
    (推导式绑定 p=[pow]、map(pow, ...)) 走私的 pow **值** 在此按实参判界 —— 与静态幂守卫
    (_guard_pow) 同水位, 且以实际值判界还额外覆盖了静态层文档化放行的动态底数。
    非 int 指数不拦 (float/complex 幂不构造大整数, 原生 OverflowError/TypeError 自然传播);
    字面指数另受静态层折叠与 |exp| 上界约束, 此处只兜高阶通路的动态量 '''
    if isinstance(base, int) and isinstance(exp, int):
        # 消息勿插值实参值: 大整数的 str 转换有 4300 位上限
        if abs(exp) > _MAX_POW_EXPONENT:
            raise SafeEvalViolation(f"pow exponent |value| > {_MAX_POW_EXPONENT} (runtime guard, static DoS mitigation)")
        if mod is None and exp >= 0 and exp * base.bit_length() > _MAX_FOLD_BITS:
            raise SafeEvalViolation("pow result too large (runtime guard, static DoS mitigation)")
    return pow(base, exp) if mod is None else pow(base, exp, mod)


_PURE_BUILTINS: dict[str, Any] = {name: getattr(builtins, name) for name in _PURE_BUILTINS_NAMES}
_PURE_BUILTINS["pow"] = _guarded_pow  # 值级守卫: 命名空间里的 pow 恒为包装版 (走私通路终将实际调用它)

#: 方法名白名单 (Attribute 访问与方法调用共用): 纯数据结构方法族; 刻意不含 format/format_map (见文件头)
_ALLOWED_METHODS = frozenset((
    "append", "extend", "insert", "pop", "remove", "clear", "index", "count", "sort", "reverse", "copy",
    "upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "join", "replace",
    "startswith", "endswith", "find", "rfind", "title", "capitalize", "isdigit", "isalpha", "isnumeric",
    "isalnum", "isspace", "islower", "isupper", "encode", "decode", "center", "ljust", "rjust", "zfill",
    "partition", "rpartition",
    "keys", "values", "items", "get", "update", "setdefault", "add", "discard", "union", "intersection", "difference",
    "real", "imag", "bit_length", "is_integer", "conjugate", "numerator", "denominator",
))

#: 语法节点允许清单 (fail-closed: 未知节点类型即拒, 未来 Python 新语法默认不可用). 纯表达式子集 (求值语义, 无语句)
_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset((
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name, ast.Attribute, ast.Subscript,
    ast.Constant, ast.IfExp, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Lambda, ast.JoinedStr, ast.FormattedValue, ast.Slice, ast.Starred,
    ast.keyword, ast.comprehension, ast.arguments, ast.arg,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
    ast.BitOr, ast.BitXor, ast.BitAnd, ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.And, ast.Or, ast.Load, ast.Store,
))

#: 静态 DoS 缓解阈值 (廉价缓解, 非硬边界 —— 见文件头已知边界)
_MAX_CODE_LENGTH = 10_000
_MAX_AST_NODES = 5_000
_MAX_POW_EXPONENT = 4_096
_MAX_REPEAT_LITERAL = 100_000_000
_MAX_FOLD_DIGITS = 10_000
_MAX_FOLD_BITS = 4 * _MAX_FOLD_DIGITS  # 折叠产物的位数预算 (~1.2 万十进制位, ~5KB)
_MAX_REPEAT_BYTES = 10 ** 9  # 序列静态尺寸预算 (按指针宽度高估, ~1GB 量级)
_SEQ_BYTES_PER_ITEM = 8


class _Huge:
    ''' 折叠哨兵: 语法上是纯数值字面量算术 (无任何动态量), 但结果超出折叠量级预算 (必为巨型) '''

    def __repr__(self) -> str:
        return "_HUGE"


_HUGE = _Huge()
_FoldResult = int | float | complex | _Huge | None

#: 折叠器支持的二元算子表 (声明式 dispatch): 算子本就按异构操作数分派, 签名按 Any 放行,
#: 非法组合 (如 float << int) 的运行期 TypeError 由调用处 except 收敛为不可折叠 —— 与裸 eval 行为一致
_FOLD_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
}


def _check_identifier(name: str) -> None:
    ''' 双下划线序列一律拒: 堵 __class__/__globals__/__builtins__/__import__ 爬链入口 '''
    if "__" in name:
        raise SafeEvalViolation(f"dunder-containing identifier is not allowed: {name!r}")


def _formats_string(node: ast.AST, literal_cache: dict[int, bool]) -> bool:
    ''' 判定 Mod 的左侧是否为字面纯的 str/bytes 模板 (如 '%200000000d' 或其拼接/分支组合):
    模板宽度/精度 ('%*d' 取自参数, '%99999999999d' 字面写死) 不可静态判界, 与 str.format 同类 fail-closed '''
    return _is_literal_pure(node, literal_cache) \
        and any(isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) for n in ast.walk(node))


def _fold_const(node: ast.AST, cache: dict[int, _FoldResult]) -> _FoldResult:
    ''' 有界数值常量折叠 (幂/移位/重复守卫共用), 带节点级 memo (守卫对每个 Mult/Pow/LShift 节点查询,
    无 memo 时左结合链的重复折叠是 O(n³) 放大器). 结果三态:
      - 数值: 折叠成功且在量级预算内
      - _HUGE: 折叠语法成立但结果超预算 (纯数值巨型, 作重复计数/幂与移位的操作数即拒)
      - None: 语法不可折叠 (名字/调用/属性/非数值常量等动态量)
    '''
    key = id(node)
    if key not in cache:
        cache[key] = _fold_uncached(node, cache)
    return cache[key]


#: 动态构造节点: 出现即表明子树含运行期才能确定的量 (字面纯度判定的否定集).
#: f-string (JoinedStr/FormattedValue) 刻意计入 —— 格式宽度 (如 f'{x:>999999999}') 是运行期格式化, 与 zfill 同类
_DYNAMIC_NODES = (ast.Name, ast.Call, ast.Attribute, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                  ast.GeneratorExp, ast.JoinedStr, ast.FormattedValue)


def _is_literal_pure(node: ast.AST, cache: dict[int, bool]) -> bool:
    ''' 字面纯度: 子树不含任何动态构造 (全是字面量与其算术/逻辑/容器组合) —— 其运行期结果为纯数据,
    尺寸/计数不可静态定界时守卫按 fail-closed 拒绝; 含动态量则属文档化放行 (bwrap 档兜底) '''
    key = id(node)
    if key not in cache:
        cache[key] = not any(isinstance(n, _DYNAMIC_NODES) for n in ast.walk(node))
    return cache[key]


def _indexed_element(node: ast.Subscript, fold_cache: dict[int, _FoldResult]) -> ast.expr | None:
    ''' 字面容器下标的静态解析 (折叠与长度两条分析线共用): 非字面 List/Tuple 容器、含 Starred 展开、
    下标不可折叠、或折叠后越界 (含负下标 —— 不静态解析) → None; 否则返回选中元素的子树 '''
    if not isinstance(node.value, (ast.List, ast.Tuple)) or any(isinstance(e, ast.Starred) for e in node.value.elts):
        return None
    index = _fold_const(node.slice, fold_cache)
    elts = node.value.elts
    if isinstance(index, int) and 0 <= index < len(elts):
        return elts[index]
    return None


def _static_len(node: ast.AST, fold_cache: dict[int, _FoldResult], len_cache: dict[int, int | None]) -> int | None:
    ''' 字面纯子树的静态序列长度上界 (memoized): 重复/拼接/切片构词的统一判界, 规则:
        字面量 = 元素数; Mult = |计数| × 序列长; Add/Sub = 两侧长之和 (拼接); Slice = 源长;
        下标 = 元素长; IfExp/BoolOp = 分支最大。动态量 / 非序列构造 / 未知宽度 → None (不可判)
    '''
    key = id(node)
    if key not in len_cache:
        len_cache[key] = _static_len_uncached(node, fold_cache, len_cache)
    return len_cache[key]


def _static_len_uncached(node: ast.AST, fold_cache: dict[int, _FoldResult], len_cache: dict[int, int | None]) -> int | None:
    def child_len(n: ast.AST) -> int | None:
        return _static_len(n, fold_cache, len_cache)

    def max_len(ns: Sequence[ast.AST]) -> int | None:
        lens: list[int] = []
        for n in ns:
            length = child_len(n)
            if length is None:
                return None
            lens.append(length)
        return max(lens)

    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return len(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            total = 0
            for e in node.elts:  # 星号展开: 展开项按其序列长计入
                if isinstance(e, ast.Starred):
                    inner = child_len(e.value)
                    if inner is None:
                        return None
                    total += inner
                else:
                    total += 1
            return total
        return len(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _fold_const(node.left, fold_cache)
        right = _fold_const(node.right, fold_cache)
        if isinstance(left, int) and not isinstance(left, bool):
            count, seq = left, node.right
        elif isinstance(right, int) and not isinstance(right, bool):
            count, seq = right, node.left
        else:
            return None
        inner = child_len(seq)
        return None if inner is None else abs(count) * inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = child_len(node.left), child_len(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return child_len(node.value)  # 切片不放大源序列
        element = _indexed_element(node, fold_cache)
        return None if element is None else child_len(element)
    if isinstance(node, ast.IfExp):
        return max_len([node.body, node.orelse])
    if isinstance(node, ast.BoolOp):
        return max_len(node.values)
    return None


def _fold_uncached(node: ast.AST, cache: dict[int, _FoldResult]) -> _FoldResult:
    ''' _fold_const 的无缓存实现: 折叠器自身不可成为放大器 —— 幂按 "底数位数 × 指数" 预估、
    左移按 "位数 + 移位量" 预估、其余算术成本与求值本身同阶 (裸折叠 10 ** 10 ** 10 会直接算出 4GB 大整数).
    覆盖: 数值字面量, 算术/位运算组合, 以及字面条件分支 (IfExp/BoolOp) 与字面容器下标 (Subscript) ——
    防止包装器把巨型数值藏进折叠盲区.
    '''
    value: int | float | complex
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # bool ⊂ int, 按其数值语义折叠 (True|10**9 即 1000000001)
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _fold_const(node.operand, cache)
        if inner is None:
            return None  # None 判定先于 _HUGE: 含动态量子树不可折叠, 哨兵保持 "纯数值巨型" 语义
        if isinstance(inner, _Huge):
            return _HUGE
        value = -inner if isinstance(node.op, ast.USub) else inner
    elif isinstance(node, ast.BinOp) and type(node.op) in _FOLD_BINOPS:
        left = _fold_const(node.left, cache)
        right = _fold_const(node.right, cache)
        if left is None or right is None:
            return None  # None 判定先于 _HUGE: 含动态量子树不可折叠, 哨兵保持 "纯数值巨型" 语义
        if isinstance(left, _Huge) or isinstance(right, _Huge):
            return _HUGE
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right >= 0 \
                and right * (left.bit_length() // 3 + 1) > _MAX_FOLD_DIGITS:  # 位数高估 (bits//3 ≥ 十进制位数), 免 str 转换 (其 4300 位上限会抛裸 ValueError)
            return _HUGE
        if isinstance(node.op, ast.LShift) and isinstance(left, int) and isinstance(right, int) \
                and left.bit_length() + right > _MAX_FOLD_BITS:  # 1 << (10 ** 9) 不可裸算
            return _HUGE
        try:
            value = _FOLD_BINOPS[type(node.op)](left, right)
        except (ArithmeticError, ValueError, OverflowError, TypeError):  # 除零/0 的负幂/浮点溢出/位运算遇 float/complex 等运行期错误 → 不可折叠
            return None
        if isinstance(value, int) and value.bit_length() > _MAX_FOLD_BITS:
            return _HUGE
        return value
    elif isinstance(node, ast.IfExp):
        test = _fold_const(node.test, cache)
        if test is None or isinstance(test, _Huge):
            return None  # _HUGE 的真值与运行期可能背离 (湮灭符: 巨型 × 0 = 0), 不可据此选支
        branch = node.body if bool(test) else node.orelse
        return _fold_const(branch, cache)
    elif isinstance(node, ast.BoolOp):
        is_or = isinstance(node.op, ast.Or)
        folded: list[_FoldResult] = []
        for value_node in node.values:
            v = _fold_const(value_node, cache)
            if v is None or isinstance(v, _Huge):  # _HUGE 真值不可判 (同上), 短路位置不可知
                return None
            folded.append(v)
            if bool(v) == is_or:  # 短路: Or 首真 / And 首假
                return v
        return folded[-1]  # Or 全假 / And 全真 → 末值
    elif isinstance(node, ast.Subscript):
        element = _indexed_element(node, cache)
        return None if element is None else _fold_const(element, cache)
    else:
        return None
    return value


def _guard_pow(base_node: ast.expr, exp_node: ast.expr, fold_cache: dict[int, _FoldResult], literal_cache: dict[int, bool]) -> None:
    ''' 幂守卫 (** 算子与 pow() 调用共用, llm-bench#12): 指数须静态折叠且 |指数| 有界;
    底数折叠为巨型、或字面纯而不可判界 → 拒; 双 int 时按位数预算判结果量级
    (单次幂即限 ~5KB, 链式相乘的各因子随之受限) '''
    base = _fold_const(base_node, fold_cache)
    exp = _fold_const(exp_node, fold_cache)
    if not isinstance(exp, (int, float, complex)) or abs(exp) > _MAX_POW_EXPONENT:
        raise SafeEvalViolation(f"pow exponent must fold to a constant with |value| <= {_MAX_POW_EXPONENT} (static DoS mitigation)")
    if isinstance(base, _Huge):  # 每层指数达标但底数复合成巨型 (如 ((2**4096)**4096)**4096) 同样拒
        raise SafeEvalViolation("pow base too large (static DoS mitigation)")
    if base is None:
        if _is_literal_pure(base_node, literal_cache):  # 字面条件/下标等折叠盲区藏底数 → fail-closed
            raise SafeEvalViolation("literal pow with statically unbounded base is rejected (fail-closed)")
    elif isinstance(base, int) and isinstance(exp, int) and exp >= 0 and exp * base.bit_length() > _MAX_FOLD_BITS:
        raise SafeEvalViolation("pow result too large (static DoS mitigation)")  # 数值结果量级判界


def _guard_pow_call(node: ast.Call, fold_cache: dict[int, _FoldResult], literal_cache: dict[int, bool]) -> None:
    ''' pow() 调用与 ** 算子同水位防护 (见文件头设计注): 星号展开/关键字/非 2-3 参静态不可判界, 一律拒 '''
    if node.keywords or not 2 <= len(node.args) <= 3 or any(isinstance(a, ast.Starred) for a in node.args):
        raise SafeEvalViolation("pow() accepts only 2 or 3 literal positional arguments (static DoS mitigation)")
    _guard_pow(node.args[0], node.args[1], fold_cache, literal_cache)


def _gate(nodes: list[ast.AST]) -> None:
    ''' 声明式白名单检查: 未知节点 / dunder 标识符 / 白名单外调用与属性 / 幂指数与序列尺寸守卫 '''
    local_names = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}  # 推导式绑定名
    callable_names = _PURE_BUILTINS.keys() | local_names
    fold_cache: dict[int, _FoldResult] = {}
    literal_cache: dict[int, bool] = {}
    len_cache: dict[int, int | None] = {}

    for node in nodes:
        if type(node) not in _ALLOWED_NODES:
            raise SafeEvalViolation(f"disallowed syntax node: {type(node).__name__} (fail-closed)")
        if isinstance(node, ast.Name):
            _check_identifier(node.id)
        elif isinstance(node, ast.Attribute):
            _check_identifier(node.attr)
            if node.attr not in _ALLOWED_METHODS:
                raise SafeEvalViolation(f"disallowed attribute access: .{node.attr}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in callable_names:
                    raise SafeEvalViolation(f"disallowed function call: {func.id}()")
                if func.id == "pow":
                    _guard_pow_call(node, fold_cache, literal_cache)
            elif isinstance(func, ast.Attribute):
                pass  # 方法调用: attr 已由 Attribute 分支对方法白名单判定
            else:
                raise SafeEvalViolation(f"disallowed call target: {ast.dump(func)}")
        elif isinstance(node, ast.arg):
            _check_identifier(node.arg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if _formats_string(node.left, literal_cache):  # 数值取模放行; 字面纯 % 格式化按 str.format 同类拒绝
                raise SafeEvalViolation("literal % formatting is rejected: width/precision is not statically bounded (fail-closed)")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _guard_pow(node.left, node.right, fold_cache, literal_cache)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
            left = _fold_const(node.left, fold_cache)
            shift = _fold_const(node.right, fold_cache)
            if isinstance(left, int) and isinstance(shift, int) and left.bit_length() + shift > _MAX_FOLD_BITS:
                raise SafeEvalViolation("left shift result too large (static DoS mitigation)")
            if isinstance(left, _Huge):
                raise SafeEvalViolation("left shift base too large (static DoS mitigation)")
            if (left is None or shift is None) and _is_literal_pure(node.left, literal_cache) \
                    and _is_literal_pure(node.right, literal_cache):  # 折叠盲区的字面移位 → fail-closed
                raise SafeEvalViolation("literal shift with statically unbounded operand is rejected (fail-closed)")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_size = _fold_const(node.left, fold_cache)
            right_size = _fold_const(node.right, fold_cache)
            left_numeric = isinstance(left_size, (int, float, complex, _Huge))
            right_numeric = isinstance(right_size, (int, float, complex, _Huge))
            if not (left_numeric and right_numeric):  # 双侧纯数值语法 = 数值乘法, 放行; 否则按计数与序列分别判界
                for size in (left_size, right_size):  # 计数侧 (数值侧) 的单位计数阈值, 独立于尺寸预算
                    if isinstance(size, _Huge) or (isinstance(size, (int, float, complex)) and abs(size) > _MAX_REPEAT_LITERAL):
                        raise SafeEvalViolation(f"repetition count exceeds {_MAX_REPEAT_LITERAL} (static DoS mitigation)")  # 消息勿插值折叠值: 大整数的 str 转换有 4300 位上限
                if left_size is None and right_size is None:
                    # 计数不可折叠 (如 [0] * (10**9 or 1)): 字面纯即拒, 含动态量则文档化放行
                    if _is_literal_pure(node.left, literal_cache) and _is_literal_pure(node.right, literal_cache):
                        raise SafeEvalViolation("literal repetition with unfoldable count is rejected (fail-closed)")
                else:
                    # 序列侧长度不可判 (如 [0, *([0] * n)] 或 f-string): 字面纯即拒, 含动态量则文档化放行
                    # (_HUGE 到达此处前必已被上方计数判界拒绝, 故 left_numeric 即 "左侧是计数")
                    seq = node.right if left_numeric else node.left
                    if _static_len(seq, fold_cache, len_cache) is None and _is_literal_pure(seq, literal_cache):
                        raise SafeEvalViolation("literal repetition with statically unbounded sequence is rejected (fail-closed)")
        seq_len = _static_len(node, fold_cache, len_cache)  # 统一尺寸预算: 重复/拼接/切片等构词路径的复合上界
        if seq_len is not None and seq_len * _SEQ_BYTES_PER_ITEM > _MAX_REPEAT_BYTES:
            raise SafeEvalViolation(f"static sequence size budget ({_MAX_REPEAT_BYTES} bytes) exceeded (static DoS mitigation)")


def safe_eval(expr: str) -> Any:
    ''' 受控求值纯表达式: 过门后在受控命名空间内执行原生 CPython 求值 (不自实现解释器, 规避 "解释器语义 bug 即安全 bug" 的整类攻击面) '''
    if len(expr) > _MAX_CODE_LENGTH:
        raise SafeEvalViolation(f"code too long: {len(expr)} > {_MAX_CODE_LENGTH}")
    tree = ast.parse(expr, mode = "eval")  # 非表达式输入 → SyntaxError (与 eval 同形, 自然传播)
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise SafeEvalViolation(f"ast too large: {len(nodes)} > {_MAX_AST_NODES}")
    try:
        _gate(nodes)
    except RecursionError:  # 超深 Mult/Pow 链的折叠递归 (~1000 项以上) → fail-closed 拒绝而非失控崩溃
        raise SafeEvalViolation("expression too deeply nested for static analysis (static DoS mitigation)")
    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": _PURE_BUILTINS})  # 代码已过门, 命名空间受控
