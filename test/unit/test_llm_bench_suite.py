# llm_bench 契约与内置 suite 的离线守护测试
#
# 职责边界: 内置 suite 加载校验 fail-fast / digest 金丝雀 (身份断代报警) / case_files 发现规则 (含 __init__.py 排除)
#           / 加载校验负例 / judge 行为 (逻辑性 bug 不改 digest, 金丝雀不报警, 须直接测)
# e2e (真实 LLM 调用) 见 test/examples/llm_bench_suite/func_simu.py

from pathlib import Path

import pytest

import ai_powered.llm_bench as apb
from ai_powered.llm_bench import Case, Instance, case_files, load_suite

_SUITES_DIR = Path(apb.__file__).parent / "suites"
_FUNC_SIMU = _SUITES_DIR / "func-simu"

# 金丝雀: func-simu 的 pinned digest —— suite 身份断代的报警器。
# 此断言变红 = suite 内容变了 (bench 身份断代), 须与 bench 侧协调 bump 依赖 rev 后有意重 pin,
# 不允许为求绿随手改 (纪律见 src/ai_powered/llm_bench/AGENTS.md)。
# 注: bench 侧旧 pin 不适用 —— 迁移使 import 行变更, 规范化源码必变 (ai-powered#12 定稿评论 "digest 不比");
#     bench 侧金丝雀在 P2 依赖切换时对本包产物重 pin。
PINNED_FUNC_SIMU_DIGEST = "8293206f38069762"


def test_builtin_func_simu_loads_and_validates() -> None:
    suite = load_suite(_FUNC_SIMU)
    assert suite.name == "func-simu"
    assert suite.rotation == "day"
    assert [c.meta.id for c in suite.cases] == [
        "func-simu/add",
        "func-simu/binary-search",
        "func-simu/extract-user-info",
        "func-simu/gen-json-data",
        "func-simu/gen-user-info",
        "func-simu/python-expression",
        "func-simu/sort-desc",
    ]
    assert suite.digest == PINNED_FUNC_SIMU_DIGEST
    assert apb.builtin_suite_digests()["func-simu"] == PINNED_FUNC_SIMU_DIGEST


def test_case_files_discovery_rules(tmp_path: Path) -> None:
    """发现规则 SSOT: *.py 排除 _ 前缀与 suite.py — 故 __init__.py 被排除 (给打包方式留自由度)"""
    (tmp_path / "_private.py").write_text("")
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "suite.py").write_text("")
    (tmp_path / "a_case.py").write_text("")
    (tmp_path / "b_case.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert [p.name for p in case_files(tmp_path)] == ["a_case.py", "b_case.py"]


def _write_neg_suite(tmp_path: Path, bad_fn: str, gen_instance: str) -> Path:
    """负例套件: 单 case bad.py — bad_fn 为任务声明 def 行, gen_instance 为单个 Instance 表达式;
    各测试的正确性依赖 "与合法 case 的唯一差异 = 被测缺陷", 由参数化增量保证"""
    d = tmp_path / "neg"
    d.mkdir()
    (d / "suite.py").write_text(
        "from ai_powered.llm_bench import SuiteMeta\n"
        "SUITE_META = SuiteMeta(name='neg', rotation='day')\n",
    )
    (d / "bad.py").write_text(
        "from ai_powered.llm_bench import Instance, case\n"
        "@case(id='neg/bad')\n"
        f"def {bad_fn}\n"
        "    '''x'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        f"    return [{gen_instance} for _ in range(n)]\n",
    )
    return d


def test_load_suite_rejects_missing_annotation(tmp_path: Path) -> None:
    d = _write_neg_suite(tmp_path, "bad(a) -> int:", "Instance(args={'a': 1}, expected=1)")
    with pytest.raises(ValueError, match="lacks type annotation"):
        load_suite(d)


def test_load_suite_rejects_expected_type_mismatch(tmp_path: Path) -> None:
    """expected 必须过返回类型 msgspec 往返 (格式契约: 'expected 通过返回类型校验')"""
    d = _write_neg_suite(tmp_path, "bad(a: int) -> int:", "Instance(args={'a': 1}, expected='not an int')")
    with pytest.raises(ValueError, match="does not conform to return type"):
        load_suite(d)


def test_load_suite_rejects_missing_without_judge(tmp_path: Path) -> None:
    """"无期望" 用 MISSING 哨兵表示且必须配 judge (格式契约第 3 条)"""
    d = _write_neg_suite(tmp_path, "bad(a: int) -> int:", "Instance(args={'a': 1})")
    with pytest.raises(ValueError, match="has no expected and case defines no judge"):
        load_suite(d)


# --- judge 行为离线测试 (逻辑性 bug 不改 digest, 金丝雀不报警, 须直接测) ---

_SUITE = load_suite(_FUNC_SIMU)


def _case(case_id: str) -> Case:
    return next(c for c in _SUITE.cases if c.meta.id == case_id)


def test_judge_gen_json_data_depth_property() -> None:
    judge = _case("func-simu/gen-json-data").judge
    assert judge is not None
    assert judge(Instance(args={"depth": 3}), {"a": {"b": [1]}})  # 深度 4 >= 3
    assert not judge(Instance(args={"depth": 3}), [])  # 空数组深度 1 (CI 模型系统性偷懒输出)
    assert not judge(Instance(args={"depth": 3}), "flat")  # 标量
    assert not judge(Instance(args={"depth": 1}), [0] * 5001)  # 规模受控


def test_judge_python_expression_eval_equality() -> None:
    judge = _case("func-simu/python-expression").judge
    assert judge is not None
    assert judge(Instance(args={"expr": "2^3"}, expected=8), "2**3")
    assert not judge(Instance(args={"expr": "2^3"}, expected=8), "2*3")  # 值不等
    assert not judge(Instance(args={"expr": "2^3"}, expected=8), "__import__('os')")  # 含下划线被拒
    assert not judge(Instance(args={"expr": "2^3"}, expected=8), 42)  # 非 str


def test_judge_gen_user_info_structure() -> None:
    # UserInfo 定义在 suite 文件内, 按路径加载取类 (与 load_suite 同款加载方式)
    from ai_powered.llm_bench.cases import _import_module

    mod = _import_module(_FUNC_SIMU / "gen_user_info.py", "test_gen_user_info_mod")
    judge = mod.judge
    user_info_cls = mod.UserInfo
    assert judge(Instance(args={}), user_info_cls(name="Alice", country="China", age="30"))
    assert not judge(Instance(args={}), user_info_cls(name="  ", country=None, age=None))  # 空白名
    assert not judge(Instance(args={}), "not a userinfo")  # 类型不符
