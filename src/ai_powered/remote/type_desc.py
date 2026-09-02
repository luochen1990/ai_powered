'''
职责: Python 类型注解 -> TypeDesc JSON 的纯函数编译器 (agent-service 动态 agent 注册 API 的输入/输出类型描述).

wire 契约来源: agent-service 服务端的 pydantic 模型 (TypeDesc ADT, kind 判别的 JSON dict), 本模块是其
client 侧单向编译器 —— 装饰期调用, fail-fast, 纯函数无 IO. 产出只携带非 None 字段
(服务端 canonical 化会剥默认值, 只带非 None 即与其 canonical 形态一致).

支持面 (完整编译规则表的正例见 test/unit/test_type_desc.py):
  标量 str/int/float/bool (精确身份匹配); Optional[T] 的全部书写形态 (maybe); list[T];
  Annotated[T, msgspec.Meta, ...] 的七类约束 (gt/ge/lt/le/pattern/min_length/max_length, 按目标 kind 分流);
  Literal[全 str 成员]; 全 str 值 Enum (wire 用成员 value, 与 msgspec 编码一致);
  product 容器 dataclass / TypedDict / NamedTuple / msgspec.Struct (含 Generic 实例化, 字段序保留声明序).

拒绝面 (UnsupportedTypeError, 报错含类型全名与人读原因 —— 装饰期 fail-fast 优于注册期/运行期的静默语义漂移):
  - dict/map: TypeDesc 无 map 原语 (融合方案 A 决策)
  - 裸 union (非 None 成员 >= 2): 服务端 sum 是 tagged 形态 ({"tag": ...}), 与本地裸 union 的 wire 形态不兼容
  - tuple / set / frozenset: 无对应原语
  - 递归 product 容器 (直接/间接自引用): 服务端 ref 环拒绝, 本地用 in-progress 类型集合检测 (栈式, 兄弟位置复用不误伤)
  - Enum 成员值非 str / Literal 混合非 str 成员 (含 None)
  - Meta 携带七类之外任何被设置字段 (multiple_of/title/description/...): 拒绝以防约束静默丢失
  - 字段名下划线开头或命中保留名单 (见 is_reserved_field_name) —— 名单是服务端 pydantic 层完整判定的
    装饰期 fail-fast 近似 (model_ 前缀一刀切 + v1 遗产名; 与服务端 pydantic 版本耦合, 极少数字段名
    仍可能注册期 422, 属已知边界)
  - 非法约束: 空区间 (gt>=lt 等, int 按整数域/float 按实数域) 与倒置的长度/数量跨度 —— 与服务端
    注册期校验同语义的前置拦截; 负值与非法 pattern 由 msgspec.Meta 构造期先行拦截 (不落本编译器)
  - 未实例化泛型 (裸 list / 裸泛型类) 与任何未识别类型 (含 Any、标量的子类; 被装饰函数所在模块启用
    from __future__ import annotations 时注解为字符串, 同样落此拒绝 —— 本地 msgspec 管线亦不支持)

缺省字段的双语境编译 (compile_type 的 defaults_omissible 参数, 2026-09 review H-1 决策):
  - 输出侧 (defaults_omissible=False, 缺省): 缺省字段按**必选**编译 —— 服务端 maybe 编译为
    "T | None = None" 且 model_dump 恒含全部字段, LLM 省略缺省字段时 wire 上是显式 null, 客户端
    Result[R] 解码对非 Optional 字段收到 null 是硬失败; 必选编译迫使 LLM 恒造值, 解码恒成功
    (代价: 缺省字段的 "省略→取缺省" 语义漂移为 "造值", 记录在此)
  - 输入侧 (defaults_omissible=True, 仅 agent.py 组装 input_spec 时): 缺省字段包 maybe —— 服务端
    输入校验对 maybe 允许省略, 恰好匹配 "调用方未显式传参时 sig.bind 不含该参数" 的 wire 形态
'''
from dataclasses import MISSING, fields as dataclass_fields, is_dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import Annotated, Any, Generic, Literal, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints, is_typeddict

import msgspec
from msgspec import NODEFAULT


class UnsupportedTypeError(ValueError):
    ''' 类型注解无法编译为 TypeDesc: 装饰期 fail-fast (优于远程注册期/运行期的静默语义漂移) '''


#: 服务端字段名判定的装饰期 fail-fast 近似 (完整判定在服务端 pydantic 层, 见 is_reserved_field_name)
#: v1 遗产方法名 —— 名单 = dir(BaseModel) 非 model_ 前缀公共成员的枚举结果 ∪ {fields} (纯 v1 遗产类属性,
#: v2 已删, 保守保留); v2 以弃用垫片形态保留其余大部分, 按历史 API 全集保守拒绝
#: (误拒成本低, 漏放成本是运行期 422; 名单生成方式: 对照服务端锁定 pydantic 版本 uv.lock 2.13.4 枚举)
_V1_LEGACY_FIELD_NAMES = frozenset({
    "json", "dict", "copy", "schema", "construct", "fields", "parse_obj", "parse_raw", "validate",
    "from_orm", "parse_file", "schema_json", "update_forward_refs",
})


def is_reserved_field_name(name: str) -> bool:
    ''' 字段名合法性 SSOT (形参名与容器字段名共用, 供 agent.py 复用):

    下划线开头 / model_ 前缀 (pydantic v2 私有命名空间, 一刀切覆盖 model_config/model_dump/
    model_validate/... 全族) / v1 遗产名 —— 三者任一即保留. 与服务端 pydantic 版本耦合:
    升级服务端 pydantic 时需回看此处 (model_ 前缀策略对 v2 全系稳健, 耦合面主要是 v1 遗产名单).
    '''
    return name.startswith("_") or name.startswith("model_") or name in _V1_LEGACY_FIELD_NAMES


#: 保留名拒绝报错的共享尾串 (容器字段名与形参名两处拼前缀复用, 策略描述与 is_reserved_field_name 同源)
RESERVED_NAME_REASON = "不合法 —— 不允许下划线开头, 且不能命中服务端保留面 (model_ 前缀 / v1 遗产名, 见 type_desc.is_reserved_field_name)"


def field_type(desc: dict[str, Any], *, omissible: bool) -> dict[str, Any]:
    ''' TypeDesc 字段条目的 wire 形态 SSOT (product 字段与 input_spec 形参条目共用):

    omissible=True 包装为 maybe (服务端校验允许省略), False 裸形态 (必选).
    '''
    return {"type": {"kind": "maybe", "element": desc} if omissible else desc}

#: msgspec.Meta 中可映射到 TypeDesc 的约束字段; 其余任何被设置字段一律拒绝 (防约束静默丢失)
_META_CONSTRAINT_FIELDS = ("gt", "ge", "lt", "le", "pattern", "min_length", "max_length")
_META_UNMAPPABLE_FIELDS = ("title", "description", "examples", "multiple_of", "tz", "extra", "extra_json_schema")

#: 各 kind 允许注入的 Meta 约束字段; list 的 min_length/max_length 在 _apply_constraints 内改名 min_items/max_items
_ALLOWED_CONSTRAINTS: Mapping[str, frozenset[str]] = {
    "int": frozenset({"gt", "ge", "lt", "le"}),
    "float": frozenset({"gt", "ge", "lt", "le"}),
    "string": frozenset({"pattern", "min_length", "max_length"}),
    "list": frozenset({"min_length", "max_length"}),
    "bool": frozenset(),
    "enum": frozenset(),
    "product": frozenset(),
    "maybe": frozenset(),
    "sum": frozenset(),  # 本编译器不产出 sum, 防御性占位
}


def compile_type(tp: Any, *, defaults_omissible: bool = False) -> dict[str, Any]:
    ''' Python 类型注解 -> TypeDesc JSON dict (纯函数, 装饰期调用, fail-fast).

    契约: 返回值只携带非 None 字段; 不可编译的类型抛 UnsupportedTypeError (报错含类型全名与人读原因).
    支持面与拒绝面见模块头注释. defaults_omissible: 容器缺省字段的语境化编译开关 (输入侧 True ->
    包 maybe 可省略; 输出侧 False -> 必选, 见模块头 "缺省字段的双语境编译").
    '''
    return _compile(tp, frozenset(), defaults_omissible = defaults_omissible)


def _compile(tp: Any, stack: frozenset[type], *, defaults_omissible: bool) -> dict[str, Any]:
    ''' 分派中枢; stack 为编译路径上的 product 容器类型集合 (命中即递归环). '''
    # Annotated (PEP 593): 以 __metadata__ 判别 —— 不用 get_origin, 其对 Annotated 的返回值跨 Python 版本不一致
    metadata = getattr(tp, "__metadata__", None)
    if metadata is not None:
        return _compile_annotated(get_args(tp)[0], metadata, tp, stack, defaults_omissible = defaults_omissible)

    # union: 恰一个非 None 成员 -> maybe (T | None / Optional[T] / Union[T, None] 统一); 裸 union 拒绝
    member = _optional_member(tp)
    if member is not None:
        return {"kind": "maybe", "element": _compile(member, stack, defaults_omissible = defaults_omissible)}

    origin = get_origin(tp)
    if origin is Union or origin is UnionType:
        raise _unsupported(tp, "裸 union (非 None 成员 >= 2) 不被支持 —— 服务端 TypeDesc 的 sum 是 tagged 形态, 与本地裸 union 的 wire 形态不兼容")

    # 标量: 精确身份匹配 (bool 先于 int 仅助读, 身份匹配无子类误伤); 子类身份无法在 wire 上表达, 走末尾拒绝
    if tp is str:
        return {"kind": "string"}
    if tp is bool:
        return {"kind": "bool"}
    if tp is int:
        return {"kind": "int"}
    if tp is float:
        return {"kind": "float"}
    if tp is NoneType or tp is None:
        raise _unsupported(tp, "None 类型仅能作为 Optional 的成员出现 (maybe 的元素不能是 None)")

    if origin is Literal:
        values = get_args(tp)
        if values and all(isinstance(v, str) for v in values):
            return {"kind": "enum", "values": list(values)}
        raise _unsupported(tp, "Literal 成员必须全为 str 且非空 (混入 None/int 等不被支持)")

    if tp is list or origin is list:
        args = get_args(tp)
        if not args:  # 裸 list / 裸 typing.List: get_args 为空
            raise _unsupported(tp, "裸 list 未参数化 —— 需要 list[T] 形态")
        (element,) = args
        return {"kind": "list", "element": _compile(element, stack, defaults_omissible = defaults_omissible)}

    # 明确拒绝的容器: 各自给出人读原因 (而非落入笼统的未识别拒绝)
    if tp is dict or origin is dict:
        raise _unsupported(tp, "dict/map 类型不被支持 —— TypeDesc 无 map 原语 (方案 A 决策), 请用 product/enum 显式建模")
    if tp is tuple or origin is tuple:
        raise _unsupported(tp, "tuple 类型不被支持 —— TypeDesc 无对应原语 (定长异构/变长同构均无)")
    if tp in (set, frozenset) or origin in (set, frozenset):
        raise _unsupported(tp, "set/frozenset 类型不被支持 —— TypeDesc 无集合原语, 序列表意请用 list[T]")

    if isinstance(tp, TypeVar):
        raise _unsupported(tp, "遇到未实例化的 TypeVar —— 泛型容器必须给出全部类型实参 (如 Pair[int, str])")

    # 类形态分派: 泛型别名剥壳到 origin (Pair[int, str] -> Pair), Enum 先于 product 容器
    cls: Any = origin if origin is not None else tp
    if not isinstance(cls, type):
        raise _unsupported(tp, "未识别的类型 —— 支持面见 ai_powered.remote.type_desc 模块头注释 (Any/特殊形态均不支持; 若被装饰函数所在模块启用了 from __future__ import annotations, 请改用运行时注解形态)")
    if issubclass(cls, Enum):
        return _enum_from(cls)
    raw_fields = _product_fields(cls)
    if raw_fields is not None:
        return _compile_product(tp, cls, raw_fields, stack, defaults_omissible = defaults_omissible)
    raise _unsupported(tp, "未识别的类型 —— 支持面见 ai_powered.remote.type_desc 模块头注释 (自定义标量子类/日期等均不支持; 若被装饰函数所在模块启用了 from __future__ import annotations, 请改用运行时注解形态)")


def _optional_member(tp: Any) -> Any | None:
    ''' 恰含一个非 None 成员的 union (Optional 的全部书写形态) -> 该成员; 其余形态返回 None.

    NoneType 用身份判别; union 的两种运行时形态 (typing.Union 与 PEP 604 types.UnionType) 均识别.
    '''
    if get_origin(tp) is Union or get_origin(tp) is UnionType:
        args = get_args(tp)
        non_none = [a for a in args if a is not NoneType]
        if len(args) == 2 and len(non_none) == 1:
            return non_none[0]
    return None


def _compile_annotated(inner: Any, metas: tuple[Any, ...], tp: Any, stack: frozenset[type], *, defaults_omissible: bool) -> dict[str, Any]:
    ''' Annotated[T, Meta, ...] -> T 的编译结果 + Meta 约束按目标 kind 分流注入.

    Optional 对称性: Annotated[Optional[S], M] 与 Optional[Annotated[S, M]] 编译结果一致 (均为
    maybe{S 的编译结果 + M}) —— 约束作用于剥掉 None 成员之后的目标.
    '''
    constraints = _merge_metas(metas, tp)
    member = _optional_member(inner)
    target = inner if member is None else member
    desc = _apply_constraints(_compile(target, stack, defaults_omissible = defaults_omissible), constraints, tp)
    return desc if member is None else {"kind": "maybe", "element": desc}


def _merge_metas(metas: tuple[Any, ...], tp: Any) -> dict[str, Any]:
    ''' 叠加合并多个 Meta 的约束 (Annotated[T, M1, M2] 自动展平为一层, 由 typing 保证).

    拒绝: 非 Meta 元数据 (如裸字符串 doc); 不可映射字段被设置 (防静默丢失); 同约束冲突值.
    '''
    merged: dict[str, Any] = {}
    for meta in metas:
        if not isinstance(meta, msgspec.Meta):
            raise _unsupported(tp, f"Annotated 的元数据必须是 msgspec.Meta 实例, 得到 {_type_name(meta)}")
        for field in _META_UNMAPPABLE_FIELDS:
            if getattr(meta, field, None) is not None:
                raise _unsupported(tp, f"msgspec.Meta 的 {field} 约束无法映射到 TypeDesc —— 拒绝以防约束静默丢失")
        for field in _META_CONSTRAINT_FIELDS:
            value = getattr(meta, field)
            if value is None:
                continue
            if field in merged and merged[field] != value:
                raise _unsupported(tp, f"多个 Meta 对约束 {field} 给出冲突值 ({merged[field]!r} vs {value!r})")
            merged[field] = value
    return merged


def _apply_constraints(desc: dict[str, Any], constraints: dict[str, Any], tp: Any) -> dict[str, Any]:
    ''' 约束按目标 kind 分流的 SSOT:

    - int/float: 仅 gt/ge/lt/le (int 节点的约束值收编为整数, 保证 wire 形态)
    - string: 仅 pattern/min_length/max_length
    - list: 仅 min_length/max_length, 改名 min_items/max_items
    - 其余 kind (bool/enum/product/maybe): 拒绝任何约束
    - 语义预检 (与服务端注册期校验同语义的前置拦截, 装饰期 fail-fast): 空区间 / 非法跨度 / 非法 pattern
    '''
    if not constraints:
        return desc
    kind = desc["kind"]
    unknown = set(constraints) - _ALLOWED_CONSTRAINTS[kind]
    if unknown:
        raise _unsupported(tp, f"约束 {sorted(unknown)} 不能作用于 {kind} 节点 (仅支持 {sorted(_ALLOWED_CONSTRAINTS[kind])})")
    if kind == "list":
        renamed = {{"min_length": "min_items", "max_length": "max_items"}[f]: v for f, v in constraints.items()}
        constraints = renamed
    if kind == "int":
        constraints = {f: _int_value(v, f, tp) for f, v in constraints.items()}
    _check_constraints_sane(kind, constraints, tp)
    if kind == "list":
        return {"kind": "list", "element": desc["element"], **constraints}
    return {"kind": kind, **constraints}


def _check_constraints_sane(kind: str, constraints: Mapping[str, Any], tp: Any) -> None:
    ''' 约束语义预检 (int/float/string/list 节点; 语义对齐服务端 adt.py 的注册期校验):

    int 按整数域判空 (开开缝隙 1 无整数, 混合边界取最紧候选), float 按实数域判空, 跨度 min<=max;
    负值与非法 pattern 由 msgspec.Meta 构造期先行拦截 (同引擎同校验), 此处不重复设防.
    '''
    if kind in ("int", "float"):
        gt, ge, lt, le = (constraints.get(f) for f in ("gt", "ge", "lt", "le"))
        if kind == "int":
            lo = max((c for c in ((gt + 1) if gt is not None else None, ge) if c is not None), default = None)
            hi = min((c for c in ((lt - 1) if lt is not None else None, le) if c is not None), default = None)
            if lo is not None and hi is not None and lo > hi:
                raise _unsupported(tp, f"约束给出空整数域: gt={gt}/ge={ge} 与 lt={lt}/le={le} 无公共整数")
        else:
            for lo_name, lo_v, hi_name, hi_v in (("gt", gt, "lt", lt), ("gt", gt, "le", le), ("ge", ge, "lt", lt)):
                if lo_v is not None and hi_v is not None and lo_v >= hi_v:
                    raise _unsupported(tp, f"约束给出空数值域: {lo_name}={lo_v} 不小于 {hi_name}={hi_v}")
            if ge is not None and le is not None and ge > le:
                raise _unsupported(tp, f"约束给出空数值域: ge={ge} 大于 le={le}")
        return
    min_name, max_name = ("min_length", "max_length") if kind == "string" else ("min_items", "max_items")
    mn, mx = constraints.get(min_name), constraints.get(max_name)
    if mn is not None and mx is not None and mn > mx:
        raise _unsupported(tp, f"约束 {min_name}={mn} 大于 {max_name}={mx}")


def _int_value(value: Any, field: str, tp: Any) -> int:
    ''' int 节点的约束值必须是整数: 整值 float 就地收编为 int (wire 上 gt: int), 其余拒绝. '''
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise _unsupported(tp, f"约束 {field}={value!r} 不是整数, 无法作用于 int 节点")


def _enum_from(cls: type[Enum]) -> dict[str, Any]:
    ''' Enum 子类 -> enum 节点: 取规范成员 (别名不计) 的 value, 声明序, 与 msgspec 的 wire 编码一致.

    规范成员间 str 值天然互异 (重复值在类创建时即成为别名), 唯一性断言仅作防御性兜底.
    '''
    values = [member.value for member in cls]
    if not values or not all(isinstance(v, str) for v in values) or len(set(values)) != len(values):
        raise _unsupported(cls, "Enum 要求非空且所有成员值为互不重复的 str (IntEnum/混合值成员不被支持)")
    return {"kind": "enum", "values": values}


def _product_fields(cls: Any) -> list[tuple[str, Any, bool]] | None:
    ''' product 容器的字段提取收口: (字段名, 注解, 可缺省?) 按声明序; 非容器返回 None.

    注解一律经 get_type_hints 解析 (覆盖 from __future__ import annotations 的字符串形态与前向引用);
    可缺省 (dataclass/Struct/NamedTuple 的 default, TypedDict 的 optional 键) 统一交由上层包 maybe.
    '''
    if is_typeddict(cls):  # 先于其余 issubclass 判定, 语义最特化
        hints = get_type_hints(cls)
        return [(name, ty, name in cls.__optional_keys__) for name, ty in hints.items()]
    if is_dataclass(cls):
        hints = get_type_hints(cls)
        return [(f.name, hints[f.name], f.default is not MISSING or f.default_factory is not MISSING) for f in dataclass_fields(cls)]
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # NamedTuple (tuple 子类收窄后静态类型即丢 _fields, 用 getattr)
        hints = get_type_hints(cls)
        field_defaults: Any = getattr(cls, "_field_defaults")
        return [(name, hints[name], name in field_defaults) for name in getattr(cls, "_fields")]
    if issubclass(cls, msgspec.Struct):
        hints = get_type_hints(cls)
        names = cls.__struct_fields__
        defaults = cls.__struct_defaults__  # 右对齐; kw_only 场景可含 NODEFAULT 占位 (default_factory 为 <factory> 哨兵)
        padded = (NODEFAULT,) * (len(names) - len(defaults)) + tuple(defaults)
        return [(name, hints[name], d is not NODEFAULT) for name, d in zip(names, padded)]
    return None


def _compile_product(tp: Any, cls: type, raw_fields: list[tuple[str, Any, bool]], stack: frozenset[type], *, defaults_omissible: bool) -> dict[str, Any]:
    ''' product 容器 -> product 节点: 泛型实例化时先对字段注解做 TypeVar 代换, 再逐字段编译.

    递归检测: cls 入栈编译字段, 出栈后兄弟位置可复用 (菱形引用不误伤); 字段名合法性在此收口.
    缺省字段: defaults_omissible=True (输入语境) 包 maybe (服务端输入校验允许省略), False (输出
    语境) 一律必选 —— 语境语义见模块头 "缺省字段的双语境编译".
    '''
    if cls in stack:
        raise _unsupported(tp, "递归类型不被远程通道支持 —— 服务端 TypeDesc 的 ref 环会被拒绝, 请消除自引用")
    args = get_args(tp)
    params = _type_parameters(cls)
    if args and len(args) != len(params):
        raise _unsupported(tp, f"泛型实参数不匹配: {_type_name(cls)} 期望 {len(params)} 个, 得到 {len(args)} 个")
    subst: Mapping[Any, Any] = dict(zip(params, args))
    fields: dict[str, Any] = {}
    for name, annotation, optional in raw_fields:
        if is_reserved_field_name(name):
            raise UnsupportedTypeError(f"{_type_name(cls)} 的字段名 {name!r} {RESERVED_NAME_REASON}")
        desc = _compile(_resolve_type(annotation, subst), stack | {cls}, defaults_omissible = defaults_omissible)
        fields[name] = field_type(desc, omissible = optional and defaults_omissible)
    return {"kind": "product", "fields": fields}


def _type_parameters(cls: type) -> tuple[Any, ...]:
    ''' 泛型参数提取: PEP 695 __type_params__ 优先, 老式 Generic[T, ...] 基类兜底 (兼容 Python 3.10 双形态). '''
    params = getattr(cls, "__type_params__", ())
    if params:
        return params
    for base in getattr(cls, "__orig_bases__", ()):
        if get_origin(base) is Generic:
            return get_args(base)
    return ()


def _resolve_type(tp: Any, subst: Mapping[Any, Any]) -> Any:
    ''' 将注解中的 TypeVar 按 subst 递归代换 (重建泛型形态, 含 Annotated/union/任意参数化别名).

    代换不完整的 TypeVar 原样保留, 由 _compile 的 TypeVar 分支给出人读拒绝. 无代换表时原样返回.
    '''
    if not subst:
        return tp
    if isinstance(tp, TypeVar):
        return subst.get(tp, tp)
    metadata = getattr(tp, "__metadata__", None)
    if metadata is not None:
        return Annotated[(_resolve_type(get_args(tp)[0], subst), *metadata)]
    if get_origin(tp) is Union or get_origin(tp) is UnionType:
        return Union[tuple(_resolve_type(a, subst) for a in get_args(tp))]
    args = get_args(tp)
    if get_origin(tp) is None or not args:
        return tp
    return get_origin(tp)[tuple(_resolve_type(a, subst) for a in args)]


def _type_name(tp: Any) -> str:
    ''' 类型全名: 类取 module.qualname; 别名/特殊形式/实例取 repr (如 "typing.Optional[int]"). '''
    if isinstance(tp, type):
        return f"{tp.__module__}.{tp.__qualname__}"
    return str(tp)


def _unsupported(tp: Any, reason: str) -> UnsupportedTypeError:
    ''' 拒绝面报错构造 SSOT: 类型全名 + 人读原因. '''
    return UnsupportedTypeError(f"{_type_name(tp)}: {reason}")
