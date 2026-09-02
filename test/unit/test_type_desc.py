'''
职责: type_desc 编译器的离线单元测试 —— Python 类型注解 -> TypeDesc JSON 的支持面 (编译规则表逐条正例,
深度解构精确断言) 与拒绝面 (UnsupportedTypeError, 报错含类型名与人读原因).

本文件启用 from __future__ import annotations: 全部 fixture 注解以字符串形态存储, 由编译器内部的
get_type_hints 解析 —— 直接覆盖 "from __future__ import annotations 场景" 的实现要点 (含前向自引用).
跨引用 fixture (递归/互递归/泛型) 定义在模块级, 保证 get_type_hints 可从模块命名空间解析.

已知权衡的回归锚: product 字段 default / TypedDict 可选键 -> 包 maybe (远程 product 字段一律必选);
Annotated[Optional[S], M] 与 Optional[Annotated[S, M]] 编译结果一致 (约束作用于剥 None 后的成员).
'''
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Generic, List, Literal, NamedTuple, NotRequired, Optional, Required, Tuple, TypedDict, TypeVar, Union

import msgspec
import pytest

from ai_powered.remote.type_desc import UnsupportedTypeError, compile_type, is_reserved_field_name


class Color(enum.Enum):
    # 标准 str 值 Enum; rojo 为 red 的同值别名 —— 规范成员不含别名, wire 值不重复
    red = "R"
    green = "G"
    rojo = "R"


class StrId(str, enum.Enum):
    # str 混入 Enum: 成员值仍为 str, 支持面内
    yes = "y"
    no = "n"


class Size(enum.IntEnum):
    # 成员值为 int: 拒绝面
    small = 1
    large = 2


class Mixed(enum.Enum):
    # 成员值类型混合: 拒绝面
    a = "x"
    b = 2


class RecursiveNode(msgspec.Struct):
    # 直接递归: 经 Optional 自引用 (栈式检测命中)
    val: int
    nxt: Optional[RecursiveNode] = None


class MutA(msgspec.Struct):
    # 互递归前半: A.b -> B
    b: MutB


class MutB(msgspec.Struct):
    # 互递归后半: B.a -> Optional[A] (间接成环)
    a: Optional[MutA] = None


class Twin(msgspec.Struct):
    # 菱形引用: 同一容器在兄弟位置复用两次 —— 非环, 栈式检测不误伤
    left: Color
    right: Color


T = TypeVar("T")


class Pair(msgspec.Struct, Generic[T]):
    # 单参泛型 Struct: 泛型实例化的基础 fixture
    first: T
    second: int


class Outer(msgspec.Struct, Generic[T]):
    # 嵌套泛型: 字段注解含参数化容器 Pair[list[T]], 考验 TypeVar 递归代换
    pair: Pair[list[T]]


@dataclass
class Point:
    # 基础 dataclass fixture: 声明序 + default 字段包 maybe
    name: str
    x: int = 0


def test_scalar_types() -> None:
    ''' 标量四原语: 精确身份匹配, 输出不含任何约束键 (None 字段省略契约) '''
    assert compile_type(str) == {"kind": "string"}
    assert compile_type(int) == {"kind": "int"}
    assert compile_type(float) == {"kind": "float"}
    assert compile_type(bool) == {"kind": "bool"}


def test_optional_all_syntaxes() -> None:
    ''' Optional 的三种书写形态 (Optional[T] / T | None / Union[T, None]) 统一编译为 maybe{T} '''
    expected = {"kind": "maybe", "element": {"kind": "string"}}
    assert compile_type(Optional[str]) == expected
    assert compile_type(str | None) == expected
    assert compile_type(Union[str, None]) == expected


def test_list_forms() -> None:
    ''' list[T] 与老式 List[T] 同译; 裸形态的拒绝另见拒绝面用例 '''
    assert compile_type(list[int]) == {"kind": "list", "element": {"kind": "int"}}
    assert compile_type(List[str]) == {"kind": "list", "element": {"kind": "string"}}


def test_string_constraints() -> None:
    ''' string 节点: pattern / min_length / max_length 同名注入 '''
    result = compile_type(Annotated[str, msgspec.Meta(pattern = r"^\d+$", min_length = 1, max_length = 5)])

    assert result == {"kind": "string", "pattern": r"^\d+$", "min_length": 1, "max_length": 5}


def test_numeric_constraints() -> None:
    ''' int / float 节点: gt/ge/lt/le 同名注入; float 约束值保持数值形态 '''
    assert compile_type(Annotated[int, msgspec.Meta(gt = 0, le = 100)]) == {"kind": "int", "gt": 0, "le": 100}
    assert compile_type(Annotated[float, msgspec.Meta(ge = 0.5, lt = 1.5)]) == {"kind": "float", "ge": 0.5, "lt": 1.5}


def test_min_length_dispatch_by_target_kind() -> None:
    ''' 约束分流 (规则表关键行为): 同一 min_length 按目标类型分流 —— string -> min_length, list -> min_items '''
    assert compile_type(Annotated[str, msgspec.Meta(min_length = 1)]) == {"kind": "string", "min_length": 1}
    assert compile_type(Annotated[list[int], msgspec.Meta(min_length = 1)]) == {"kind": "list", "element": {"kind": "int"}, "min_items": 1}
    assert compile_type(Annotated[list[str], msgspec.Meta(min_length = 1, max_length = 10)]) == {
        "kind": "list", "element": {"kind": "string"}, "min_items": 1, "max_items": 10,
    }


def test_meta_stacking_and_idempotent_repeat() -> None:
    ''' 多 Meta 叠加 (Annotated[T, M1, M2]) 合并注入; 同值重复设置视为幂等 '''
    result = compile_type(Annotated[str, msgspec.Meta(min_length = 1), msgspec.Meta(max_length = 10)])

    assert result == {"kind": "string", "min_length": 1, "max_length": 10}
    assert compile_type(Annotated[int, msgspec.Meta(ge = 0), msgspec.Meta(ge = 0)]) == {"kind": "int", "ge": 0}


def test_annotated_optional_symmetry() -> None:
    ''' Optional 对称性: Annotated[Optional[S], M] 与 Optional[Annotated[S, M]] 编译结果一致 (maybe{S+M}) '''
    expected = {"kind": "maybe", "element": {"kind": "string", "min_length": 1}}

    assert compile_type(Annotated[Optional[str], msgspec.Meta(min_length = 1)]) == expected
    assert compile_type(Optional[Annotated[str, msgspec.Meta(min_length = 1)]]) == expected


def test_constraint_through_optional_list() -> None:
    ''' 约束穿透 Optional 作用于成员: Annotated[Optional[list[int]], M(min_length)] -> maybe{list{min_items}} '''
    result = compile_type(Annotated[Optional[list[int]], msgspec.Meta(min_length = 1)])

    assert result == {"kind": "maybe", "element": {"kind": "list", "element": {"kind": "int"}, "min_items": 1}}


def test_integral_float_constraint_coerced_on_int() -> None:
    ''' int 节点的整值 float 约束就地收编为 int (wire 上 gt: int 形态) '''
    assert compile_type(Annotated[int, msgspec.Meta(gt = 2.0)]) == {"kind": "int", "gt": 2}


def test_literal_of_str() -> None:
    ''' Literal[全 str, 非空] -> enum{values}, 成员序保留书写序 '''
    assert compile_type(Literal["low", "medium", "high"]) == {"kind": "enum", "values": ["low", "medium", "high"]}


def test_enum_values_are_member_values() -> None:
    ''' Enum -> enum{values}: 取成员 value (非 name, 与 msgspec wire 编码一致), 声明序, 别名成员不产生重复值 '''
    assert compile_type(Color) == {"kind": "enum", "values": ["R", "G"]}
    assert compile_type(StrId) == {"kind": "enum", "values": ["y", "n"]}


def test_dataclass_product() -> None:
    ''' dataclass -> product: 字段序保留声明序; 缺省字段双语境 —— 输出语境 (缺省) 必选, 输入语境包 maybe '''
    assert compile_type(Point) == {
        "kind": "product",
        "fields": {
            "name": {"type": {"kind": "string"}},
            "x": {"type": {"kind": "int"}},
        },
    }
    assert compile_type(Point, defaults_omissible = True) == {
        "kind": "product",
        "fields": {
            "name": {"type": {"kind": "string"}},
            "x": {"type": {"kind": "maybe", "element": {"kind": "int"}}},
        },
    }


def test_frozen_dataclass_product() -> None:
    ''' frozen dataclass 同为 product (含 frozen 形态回归) '''

    @dataclass(frozen = True)
    class FrozenPoint:
        x: int
        y: int

    assert compile_type(FrozenPoint) == {
        "kind": "product",
        "fields": {"x": {"type": {"kind": "int"}}, "y": {"type": {"kind": "int"}}},
    }


def test_dataclass_factory_default_dual_context() -> None:
    ''' dataclass 字段 default_factory 同为 "带 default" -> 双语境分流 (输出必选 / 输入 maybe) '''

    @dataclass
    class WithFactory:
        items: List[int] = field(default_factory = list)

    assert compile_type(WithFactory) == {
        "kind": "product",
        "fields": {"items": {"type": {"kind": "list", "element": {"kind": "int"}}}},
    }
    assert compile_type(WithFactory, defaults_omissible = True) == {
        "kind": "product",
        "fields": {"items": {"type": {"kind": "maybe", "element": {"kind": "list", "element": {"kind": "int"}}}}},
    }


def test_typeddict_total_true() -> None:
    ''' TypedDict total=True: 全部字段必选, 不包 maybe '''

    class TD(TypedDict):
        name: str
        count: int

    assert compile_type(TD) == {
        "kind": "product",
        "fields": {"name": {"type": {"kind": "string"}}, "count": {"type": {"kind": "int"}}},
    }


def test_typeddict_total_false() -> None:
    ''' TypedDict total=False: 全部字段包 maybe; Required[...] 键恢复必选

    NOTE: 功能语法 (而非 class 语法) —— 本文件启用的 from __future__ import annotations 使 class 语法的
    Required 注解延迟为字符串, CPython TypedDict 元类对字符串形态不做 required/optional 分类 (上游限制),
    功能语法的字段值是运行时真实对象, 分类正确; 编译器消费 __optional_keys__, 输入契约由调用方保证.
    '''
    TD = TypedDict("TD", {"a": int, "b": Required[str]}, total = False)

    assert compile_type(TD) == {
        "kind": "product",
        "fields": {
            "a": {"type": {"kind": "int"}},
            "b": {"type": {"kind": "string"}},
        },
    }
    assert compile_type(TD, defaults_omissible = True) == {
        "kind": "product",
        "fields": {
            "a": {"type": {"kind": "maybe", "element": {"kind": "int"}}},
            "b": {"type": {"kind": "string"}},
        },
    }


def test_typeddict_not_required() -> None:
    ''' total=True 中的 NotRequired 键双语境分流, 其余必选 (功能语法缘由见上) '''
    TD = TypedDict("TD", {"x": int, "y": NotRequired[str]})

    assert compile_type(TD) == {
        "kind": "product",
        "fields": {
            "x": {"type": {"kind": "int"}},
            "y": {"type": {"kind": "string"}},
        },
    }
    assert compile_type(TD, defaults_omissible = True) == {
        "kind": "product",
        "fields": {
            "x": {"type": {"kind": "int"}},
            "y": {"type": {"kind": "maybe", "element": {"kind": "string"}}},
        },
    }


def test_namedtuple_product() -> None:
    ''' NamedTuple -> product: 字段序保留声明序; 带 default 字段双语境分流 '''

    class NP(NamedTuple):
        a: int
        b: str = "x"

    assert compile_type(NP) == {
        "kind": "product",
        "fields": {
            "a": {"type": {"kind": "int"}},
            "b": {"type": {"kind": "string"}},
        },
    }
    assert compile_type(NP, defaults_omissible = True) == {
        "kind": "product",
        "fields": {
            "a": {"type": {"kind": "int"}},
            "b": {"type": {"kind": "maybe", "element": {"kind": "string"}}},
        },
    }


def test_msgspec_struct_product() -> None:
    ''' msgspec.Struct -> product: default 字段 (含 default_factory) 双语境分流 '''

    class Item(msgspec.Struct):
        label: str
        tags: List[str] = msgspec.field(default_factory = list)

    assert compile_type(Item) == {
        "kind": "product",
        "fields": {
            "label": {"type": {"kind": "string"}},
            "tags": {"type": {"kind": "list", "element": {"kind": "string"}}},
        },
    }
    assert compile_type(Item, defaults_omissible = True) == {
        "kind": "product",
        "fields": {
            "label": {"type": {"kind": "string"}},
            "tags": {"type": {"kind": "maybe", "element": {"kind": "list", "element": {"kind": "string"}}}},
        },
    }


def test_generic_struct_instantiation() -> None:
    ''' 泛型 Struct 实例化: TypeVar 按实参代换 (Pair[int] 的 first: T -> int; second 为具体类型不受影响) '''
    result = compile_type(Pair[int])

    assert result == {
        "kind": "product",
        "fields": {"first": {"type": {"kind": "int"}}, "second": {"type": {"kind": "int"}}},
    }


def test_generic_struct_nested_substitution() -> None:
    ''' 嵌套泛型: Outer[bool] 的字段 Pair[list[T]] 需递归代换 T -> bool '''
    result = compile_type(Outer[bool])

    assert result == {
        "kind": "product",
        "fields": {
            "pair": {
                "type": {
                    "kind": "product",
                    "fields": {
                        "first": {"type": {"kind": "list", "element": {"kind": "bool"}}},
                        "second": {"type": {"kind": "int"}},
                    },
                }
            }
        },
    }

def test_nested_combination() -> None:
    ''' 深嵌套组合: Optional[list[dataclass]] 与 list[Optional[str]] (元素级 maybe);
    嵌套容器的缺省字段同样按顶层语境分流 (输出必选 / 输入 maybe) '''
    assert compile_type(Optional[List[Point]]) == {
        "kind": "maybe",
        "element": {
            "kind": "list",
            "element": {
                "kind": "product",
                "fields": {
                    "name": {"type": {"kind": "string"}},
                    "x": {"type": {"kind": "int"}},
                },
            },
        },
    }
    assert compile_type(Optional[List[Point]], defaults_omissible = True) == {
        "kind": "maybe",
        "element": {
            "kind": "list",
            "element": {
                "kind": "product",
                "fields": {
                    "name": {"type": {"kind": "string"}},
                    "x": {"type": {"kind": "maybe", "element": {"kind": "int"}}},
                },
            },
        },
    }
    assert compile_type(List[Optional[str]]) == {"kind": "list", "element": {"kind": "maybe", "element": {"kind": "string"}}}


def test_sibling_reuse_not_recursion() -> None:
    ''' 菱形引用: 同一类型在兄弟位置复用 (Twin.left/right 同为 Color) 不触发递归拒绝 '''
    result = compile_type(Twin)

    assert result == {
        "kind": "product",
        "fields": {
            "left": {"type": {"kind": "enum", "values": ["R", "G"]}},
            "right": {"type": {"kind": "enum", "values": ["R", "G"]}},
        },
    }


def test_reject_dict_has_no_map_primitive() -> None:
    ''' dict/Dict 全形态拒绝: TypeDesc 无 map 原语 (方案 A 决策), 报错含类型名与原因 '''
    with pytest.raises(UnsupportedTypeError, match = re.escape("dict[str, int]")):
        compile_type(dict[str, int])
    with pytest.raises(UnsupportedTypeError, match = "typing.Dict"):
        compile_type(Dict)
    with pytest.raises(UnsupportedTypeError, match = "map"):
        compile_type(dict)


def test_reject_bare_union() -> None:
    ''' 裸 union (非 None 成员 >= 2) 拒绝: 服务端 sum 是 tagged 形态, wire 不兼容 '''
    with pytest.raises(UnsupportedTypeError, match = re.escape("int | str")):
        compile_type(int | str)
    with pytest.raises(UnsupportedTypeError, match = "union"):
        compile_type(Union[int, str, None])


def test_reject_tuple() -> None:
    ''' tuple 全形态 (定长异构/变长/裸) 拒绝: 无对应原语 '''
    with pytest.raises(UnsupportedTypeError, match = re.escape("tuple[int, str]")):
        compile_type(tuple[int, str])
    with pytest.raises(UnsupportedTypeError, match = "tuple"):
        compile_type(tuple[int, ...])
    with pytest.raises(UnsupportedTypeError, match = "typing.Tuple"):
        compile_type(Tuple)


def test_reject_recursive_types() -> None:
    ''' 递归检测: 直接自引用与互递归均拒绝 (in-progress 集合), 报错含 "递归" 指引 '''
    with pytest.raises(UnsupportedTypeError, match = "RecursiveNode.*递归"):
        compile_type(RecursiveNode)
    with pytest.raises(UnsupportedTypeError, match = "MutA.*递归"):
        compile_type(MutA)
    with pytest.raises(UnsupportedTypeError, match = "MutB.*递归"):
        compile_type(MutB)


def test_reject_non_str_enum_members() -> None:
    ''' Enum 成员值非 str (IntEnum) 或混合类型: 拒绝, 报错含类型名 '''
    with pytest.raises(UnsupportedTypeError, match = "Size"):
        compile_type(Size)
    with pytest.raises(UnsupportedTypeError, match = "Mixed"):
        compile_type(Mixed)


def test_reject_literal_mixed_members() -> None:
    ''' Literal 混合非 str 成员 (含 None): 拒绝 '''
    with pytest.raises(UnsupportedTypeError, match = re.escape("Literal['a', 1]")):
        compile_type(Literal["a", 1])
    with pytest.raises(UnsupportedTypeError, match = "Literal"):
        compile_type(Literal["a", None])


def test_reject_set_and_frozenset() -> None:
    ''' set/frozenset 拒绝: 无集合原语 '''
    with pytest.raises(UnsupportedTypeError, match = re.escape("set[int]")):
        compile_type(set[int])
    with pytest.raises(UnsupportedTypeError, match = "frozenset"):
        compile_type(frozenset[int])


def test_reject_unrecognized_types() -> None:
    ''' 未识别类型 (Any / 任意第三方类 / 标量子类 / NoneType) 拒绝, 报错含类型名 '''

    class MyStr(str):
        pass

    import datetime

    with pytest.raises(UnsupportedTypeError, match = "typing.Any"):
        compile_type(Any)
    with pytest.raises(UnsupportedTypeError, match = "datetime.datetime"):
        compile_type(datetime.datetime)
    with pytest.raises(UnsupportedTypeError, match = "MyStr"):
        compile_type(MyStr)
    with pytest.raises(UnsupportedTypeError, match = "None"):
        compile_type(type(None))


def test_reject_bare_list() -> None:
    ''' 裸 list / 裸 List 未参数化: 拒绝 (泛型别名未实例化) '''
    with pytest.raises(UnsupportedTypeError, match = "裸 list"):
        compile_type(list)
    with pytest.raises(UnsupportedTypeError, match = "typing.List"):
        compile_type(List)


def test_reject_underscore_field_name() -> None:
    ''' 字段名下划线开头: 装饰期 fail-fast 拒绝 '''

    @dataclass
    class HasPrivate:
        _private: int

    with pytest.raises(UnsupportedTypeError, match = "HasPrivate.*_private"):
        compile_type(HasPrivate)


def test_reject_reserved_field_name() -> None:
    ''' 字段名命中服务端保留名单 (客户端 fail-fast 子集): 拒绝 '''

    @dataclass
    class HasReserved:
        schema: int

    class BadStruct(msgspec.Struct):
        json: str

    with pytest.raises(UnsupportedTypeError, match = "HasReserved.*schema"):
        compile_type(HasReserved)
    with pytest.raises(UnsupportedTypeError, match = "BadStruct.*json"):
        compile_type(BadStruct)


def test_reject_unmappable_meta_fields() -> None:
    ''' Meta 携带七类之外的被设置字段 (multiple_of 等约束 / title/description 等元信息): 拒绝防静默丢失 '''
    with pytest.raises(UnsupportedTypeError, match = "multiple_of"):
        compile_type(Annotated[int, msgspec.Meta(ge = 0, multiple_of = 5)])
    with pytest.raises(UnsupportedTypeError, match = "description"):
        compile_type(Annotated[str, msgspec.Meta(description = "doc")])
    with pytest.raises(UnsupportedTypeError, match = "title"):
        compile_type(Annotated[str, msgspec.Meta(title = "t")])


def test_reject_constraint_kind_mismatch() -> None:
    ''' 约束与目标 kind 不匹配 (pattern 作用 int / gt 作用 string / 约束作用 bool 与 product): 拒绝 '''

    @dataclass
    class Plain:
        x: int

    with pytest.raises(UnsupportedTypeError, match = "pattern.*int"):
        compile_type(Annotated[int, msgspec.Meta(pattern = "x")])
    with pytest.raises(UnsupportedTypeError, match = "gt.*string"):
        compile_type(Annotated[str, msgspec.Meta(gt = 1)])
    with pytest.raises(UnsupportedTypeError, match = "ge.*bool"):
        compile_type(Annotated[bool, msgspec.Meta(ge = 1)])
    with pytest.raises(UnsupportedTypeError, match = "pattern.*list"):
        compile_type(Annotated[list[int], msgspec.Meta(pattern = "x")])
    with pytest.raises(UnsupportedTypeError, match = "min_length.*product"):
        compile_type(Annotated[Plain, msgspec.Meta(min_length = 1)])


def test_reject_conflicting_metas() -> None:
    ''' 多 Meta 对同一约束给出不同值: 拒绝 (语义二义) '''
    with pytest.raises(UnsupportedTypeError, match = "冲突值"):
        compile_type(Annotated[int, msgspec.Meta(ge = 0), msgspec.Meta(ge = 1)])


def test_reject_fractional_constraint_on_int() -> None:
    ''' int 节点的非整值约束 (gt=1.5): 拒绝 (wire 上 int 节点约束须为整数) '''
    with pytest.raises(UnsupportedTypeError, match = "1.5"):
        compile_type(Annotated[int, msgspec.Meta(gt = 1.5)])


def test_reject_unparametrized_generic() -> None:
    ''' 未实例化泛型 (裸泛型类, 字段含裸 TypeVar): 拒绝 '''
    with pytest.raises(UnsupportedTypeError, match = "TypeVar"):
        compile_type(Pair)


def test_reject_non_meta_annotation_metadata() -> None:
    ''' Annotated 元数据非 msgspec.Meta (如裸字符串 doc): 拒绝 '''
    with pytest.raises(UnsupportedTypeError, match = "msgspec.Meta"):
        compile_type(Annotated[str, "some doc"])


def test_reserved_field_name_rules() -> None:
    ''' 保留面判定 (M-2 规则): model_ 前缀一刀切 / v1 遗产名 / 下划线开头; 普通名与 "dump"/"model_url" 之外的常规名放行 '''
    assert is_reserved_field_name("model_config")
    assert is_reserved_field_name("model_dump")
    assert is_reserved_field_name("model_validate")
    assert is_reserved_field_name("model_url")  # model_ 前缀保守误拒 (注释记录的已知代价)
    assert is_reserved_field_name("json")
    assert is_reserved_field_name("schema")
    assert is_reserved_field_name("_private")
    assert not is_reserved_field_name("name")
    assert not is_reserved_field_name("dump")
    assert not is_reserved_field_name("model")


def test_reject_empty_numeric_interval() -> None:
    ''' 非法约束预检 (M-3): 空区间装饰期拒绝 —— int 按整数域 (开开缝隙 1 无整数), float 按实数域 '''
    with pytest.raises(UnsupportedTypeError, match = "空整数域"):
        compile_type(Annotated[int, msgspec.Meta(gt = 0, lt = 1)])
    with pytest.raises(UnsupportedTypeError, match = "空整数域"):  # 混合边界取最紧候选: gt=0, le=0 无公共整数
        compile_type(Annotated[int, msgspec.Meta(gt = 0, le = 0)])
    with pytest.raises(UnsupportedTypeError, match = "空数值域"):
        compile_type(Annotated[float, msgspec.Meta(ge = 1.0, lt = 1.0)])
    # 合法边界不受影响 (回归锚)
    assert compile_type(Annotated[int, msgspec.Meta(ge = 0, le = 1)]) == {"kind": "int", "ge": 0, "le": 1}


def test_reject_invalid_span_and_pattern() -> None:
    ''' 非法约束预检: 倒置跨度装饰期拒绝 (对齐服务端注册期校验); 负值与非法 pattern 由
    msgspec.Meta 构造期先行拒绝 (min_length=-1 / pattern="(" 在 Meta(...) 即抛), 不落本预检 '''
    with pytest.raises(UnsupportedTypeError, match = "大于"):
        compile_type(Annotated[str, msgspec.Meta(min_length = 10, max_length = 1)])
    with pytest.raises(UnsupportedTypeError, match = "大于"):
        compile_type(Annotated[list[int], msgspec.Meta(min_length = 3, max_length = 2)])  # list 侧改名后同检


def test_reject_string_annotation_with_hint() -> None:
    ''' L-5 回归锚: 字符串注解 (PEP 563) 拒绝时报错携带修复指引 '''
    with pytest.raises(UnsupportedTypeError, match = "from __future__ import annotations"):
        compile_type("int")
