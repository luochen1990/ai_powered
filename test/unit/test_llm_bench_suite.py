# llm_bench 契约与内置 suite 的离线守护测试
#
# 职责边界: 内置 suite 加载校验 fail-fast / digest 金丝雀 (身份断代报警) / case_files 发现规则 (含 __init__.py 排除)
#           / 加载校验负例 / judge 行为 (逻辑性 bug 不改 digest, 金丝雀不报警, 须直接测)
# e2e (真实 LLM 调用) 见 test/examples/llm_bench_suite/func_simu.py

import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ai_powered.llm_bench as apb
from ai_powered.llm_bench import Case, Instance, case_files, load_suite

_SUITES_DIR = Path(apb.__file__).parent / "suites"
_FUNC_SIMU = _SUITES_DIR / "func-simu"

# 金丝雀: func-simu 的 pinned digest —— suite 身份断代的报警器。
# 此断言变红 = suite 内容变了 (bench 身份断代), 须与 bench 侧协调 bump 依赖 rev 后有意重 pin,
# 不允许为求绿随手改 (纪律见 src/ai_powered/llm_bench/AGENTS.md)。
# 格式: 分层 "<prompt>*<distribution>:<judging>" (各段 16 hex); 前缀比较 = 粒度选择
# 历史: 8293206f38069762 (P1 迁移, 单级) → 62e67dd03c540281 (两级化+name/rotation 出哈希)
#       → 7b906e2e01c394f2 (@case 元数据剔除) → 分层三段 (段级联; 均为算法断代非内容断代)
#       → 07696cf4d703b064*7fb424258ad23a0b:9701dd524a92f3fd (分层格式首版)
# 重 pin (llm-bench#12 上游半, 内容断代·仅判定段): python_expression judge 委托 utils.safe_eval
#       (DoS 静态缓解) — 题型/分布段与其余 6 case 不变; bench 侧 bump rev 后重 pin 吸收
PINNED_FUNC_SIMU_DIGEST = "07696cf4d703b064*7fb424258ad23a0b:533b410397ba5934"
PINNED_CASE_DIGESTS = {
    "func-simu/add": "a4cf80aa48805fc8*97c515322ac7c10a:03ea3778f24e91d8",
    "func-simu/binary-search": "359eebb244b2c658*f963852e3536a136:e3693aeeed72a44c",
    "func-simu/extract-user-info": "c0fe2f17a847f08b*ebaba4561892bf2a:222369c8936cbba4",
    "func-simu/gen-json-data": "a46ac50af8a9287a*803eabba35ea0e4c:e392d3680e627792",
    "func-simu/gen-user-info": "b063e29a556ebf95*5f0803cbc88e71b0:cd8bd2d0900b1843",
    # 判定段 1644e4342bd833b5 → 7246319b7f09ec3d (llm-bench#12: judge 委托 utils.safe_eval)
    "func-simu/python-expression": "b980a939e4542ebe*6a6b4ff4af611ee5:7246319b7f09ec3d",
    "func-simu/sort-desc": "1d49eeb6a2e64682*d5d4abf62ad38e97:c438ab044700d112",
}


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
    # 两级指纹: per-case digest 与独立计算一致 (load 与 digest 同源); 分层串格式 (题型*分布:判定)
    assert {c.meta.id: str(c.digest) for c in suite.cases} == PINNED_CASE_DIGESTS
    assert {k: str(v) for k, v in apb.suite_digests(_FUNC_SIMU).items()} == PINNED_CASE_DIGESTS


def test_case_files_discovery_rules(tmp_path: Path) -> None:
    """发现规则 SSOT: *.py 排除 _ 前缀与 suite.py — 故 __init__.py 被排除 (给打包方式留自由度)"""
    (tmp_path / "_private.py").write_text("")
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "suite.py").write_text("")
    (tmp_path / "a_case.py").write_text("")
    (tmp_path / "b_case.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert [p.name for p in case_files(tmp_path)] == ["a_case.py", "b_case.py"]


# --- 共享脚手架: 微型套件 / 负例套件构造 (digest 性质测试与 loader 负例测试共用) ---

_GOOD_CASE = (
    "from ai_powered.llm_bench import Instance, case\n"
    "@case(id='neg/bad')\n"
    "def bad(a: int) -> int:\n"
    "    '''x'''\n"
    "    ...\n"
    "def gen(rng, n):\n"
    "    return [Instance(args={'a': 1}, expected=1) for _ in range(n)]\n"
)


def _mini_suite(
    base: Path,
    name: str = "neg",
    case_body: str = _GOOD_CASE,
    filename: str = "bad.py",
    rotation: str = "day",
) -> Path:
    """微型套件: base/<name>/ 下一个 suite.py + 一个 case 文件 (name 同时作目录名与 meta.name — load_suite 校验二者一致)"""
    d = base / name
    d.mkdir(parents=True)
    (d / "suite.py").write_text(f"from ai_powered.llm_bench import SuiteMeta\nSUITE_META = SuiteMeta(name='{name}', rotation='{rotation}')\n")
    (d / filename).write_text(case_body)
    return d


def _write_neg_suite(tmp_path: Path, bad_fn: str, gen_instance: str) -> Path:
    """负例套件: 单 case bad.py — bad_fn 为任务声明 def 行, gen_instance 为单个 Instance 表达式;
    各测试的正确性依赖 "与合法 case 的唯一差异 = 被测缺陷", 由参数化增量保证"""
    body = (
        "from ai_powered.llm_bench import Instance, case\n"
        "@case(id='neg/bad')\n"
        f"def {bad_fn}\n"
        "    '''x'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        f"    return [{gen_instance} for _ in range(n)]\n"
    )
    return _mini_suite(tmp_path, case_body=body)


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


# --- digest 两级化性质测试 (算法契约的行为化, llm-bench#9 P3a) ---

def test_digest_insensitive_to_suite_name_and_rotation(tmp_path: Path) -> None:
    """分布身份与采样点身份正交: name/rotation 参与种子派生但不进内容指纹 (改名不断代)"""
    a = _mini_suite(tmp_path, "a")
    b = _mini_suite(tmp_path, "b")  # 目录名与 meta.name 一致 (load_suite 校验), case 相同
    assert apb.suite_digest(a) == apb.suite_digest(b)
    rotated = _mini_suite(tmp_path, "c", rotation="week")
    assert apb.suite_digest(a) == apb.suite_digest(rotated)


def test_digest_insensitive_to_case_filename(tmp_path: Path) -> None:
    """聚合按 case id 排序 (非文件名): 重命名 case 文件不断代 (id 是题号, 改 id 才是换题)"""
    a = _mini_suite(tmp_path, "a", filename="zz.py")
    b = _mini_suite(tmp_path, "b", filename="aa.py")
    assert apb.suite_digest(a) == apb.suite_digest(b)


def test_digest_insensitive_to_case_metadata(tmp_path: Path) -> None:
    """@case 元数据是标签层非测量内容: difficulty/source/description 改动不断代"""
    base = _GOOD_CASE.replace(
        "@case(id='neg/bad')",
        "@case(id='neg/bad', difficulty='easy', source='upstream:x@rev1', description='d1')",
    )
    relabeled = _GOOD_CASE.replace(
        "@case(id='neg/bad')",
        "@case(id='neg/bad', difficulty='hard', source='upstream:x@rev2', description='d2')",
    )
    a = _mini_suite(tmp_path, "a", case_body=base)
    b = _mini_suite(tmp_path, "b", case_body=relabeled)
    assert apb.suite_digests(a) == apb.suite_digests(b)
    assert apb.suite_digest(a) == apb.suite_digest(b)


def test_case_id_change_moves_aggregation_key_only(tmp_path: Path) -> None:
    """改 id = 换题号而非换题: case_digest (题本身) 不变, suite_digest (聚合键) 变"""
    a = _mini_suite(tmp_path, "a")
    b = _mini_suite(tmp_path, "b", case_body=_GOOD_CASE.replace("neg/bad", "neg/renamed"))
    da, db = apb.suite_digests(a), apb.suite_digests(b)
    assert da["neg/bad"] == db["neg/renamed"]  # 同题
    assert apb.suite_digest_from_cases(da) != apb.suite_digest_from_cases(db)  # 不同题号


def test_digest_case_change_is_surgical(tmp_path: Path) -> None:
    """单 case 变 → 该 case 各段指纹变 + suite 合串变, 其余 case 指纹不变 (部分变更归因)"""
    changed_body = _GOOD_CASE.replace("expected=1", "expected=2")
    a, b = _mini_suite(tmp_path, "a"), _mini_suite(tmp_path, "b", case_body=changed_body)
    da, db = apb.suite_digests(a), apb.suite_digests(b)
    assert str(da["neg/bad"]) != str(db["neg/bad"])
    assert apb.suite_digest_from_cases(da) != apb.suite_digest_from_cases(db)


# --- 分层语义测试 (题型*分布:判定; 前缀比较 = 粒度选择) ---

_JUDGE_CASE = (
    "from ai_powered.llm_bench import Instance, case\n"
    "@case(id='neg/bad')\n"
    "def bad(a: int) -> int:\n"
    "    '''x'''\n"
    "    ...\n"
    "def gen(rng, n):\n"
    "    return [Instance(args={'a': 1}) for _ in range(n)]\n"
    "def judge(instance, output):\n"
    "    return output == 1\n"
)


def _layers(d: Path) -> tuple[str, str, str]:
    (layered,) = apb.suite_digests(d).values()
    return layered.prompt, layered.distribution, layered.judging


def _layer_pair(tmp_path: Path, body: str, changed: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """基线/变体两微型套件的分层指纹 — 分层语义测试共用 (被测内容在各测试的 body 差异与断言里)"""
    return _layers(_mini_suite(tmp_path, "a", case_body=body)), _layers(_mini_suite(tmp_path, "b", case_body=changed))


def test_layer_judge_change_keeps_prompt_and_distribution(tmp_path: Path) -> None:
    """judge 修复 (改卷): 题型段与分布段不变, 仅判定段变 —— 前缀比较口径: 仍同题库, 不再统计可比"""
    changed = _JUDGE_CASE.replace("output == 1", "output == 2")
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, _JUDGE_CASE, changed)
    assert (pa, da) == (pb, db)  # 题型与分布不变
    assert ja != jb  # 判定语义变


def test_layer_gen_change_keeps_prompt(tmp_path: Path) -> None:
    """gen 变更 (换题库): 题型段不变, 分布段起变 (级联传导至判定段)"""
    changed = _JUDGE_CASE.replace("args={'a': 1}", "args={'a': 2}")
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, _JUDGE_CASE, changed)
    assert pa == pb  # 题型不变
    assert da != db  # 分布变
    assert ja != jb  # 级联: 判定段以分布段为前缀, 随之变


def test_layer_prompt_change_cascades_all(tmp_path: Path) -> None:
    """题型变更 (docstring 是 prompt 一部分): 题型段变, 级联传导至全部后段"""
    changed = _JUDGE_CASE.replace("'''x'''", "'''y'''")
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, _JUDGE_CASE, changed)
    assert pa != pb and da != db and ja != jb  # 级联全变


def test_layer_shared_symbol_goes_to_coarsest(tmp_path: Path) -> None:
    """多层共用符号归最粗锚点层: 签名默认值 (prompt) 与 judge 共用的常量, 改它 → 题型段变 (级联全变)"""
    shared = (
        "from ai_powered.llm_bench import Instance, case\n"
        "LIMIT = 1\n"
        "@case(id='neg/bad')\n"
        "def bad(a: int = LIMIT) -> int:\n"
        "    '''x'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        "    return [Instance(args={}) for _ in range(n)]\n"
        "def judge(instance, output):\n"
        "    return output == LIMIT\n"
    )
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, shared, shared.replace("LIMIT = 1", "LIMIT = 2"))
    assert pa != pb and da != db and ja != jb  # 归题型层, 级联传导


def test_layer_gen_only_symbol_stays_in_distribution(tmp_path: Path) -> None:
    """仅分布层引用的符号: 改它题型段不变 (gen 域常量调整不断代题型)"""
    shared = (
        "from ai_powered.llm_bench import Instance, case\n"
        "SPAN = 5\n"
        "@case(id='neg/bad')\n"
        "def bad(a: int) -> int:\n"
        "    '''x'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        "    return [Instance(args={'a': rng.randrange(SPAN)}) for _ in range(n)]\n"
        "def judge(instance, output):\n"
        "    return output == 1\n"
    )
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, shared, shared.replace("SPAN = 5", "SPAN = 10"))
    assert pa == pb  # 题型不变
    assert da != db  # 分布变


def test_layer_judge_only_symbol_stays_in_judging(tmp_path: Path) -> None:
    """仅判定层引用的符号: 改它题型/分布段不变 ("judge 修复不断代题库"的核心场景)"""
    shared = (
        "from ai_powered.llm_bench import Instance, case\n"
        "THRESHOLD = 1\n"
        "@case(id='neg/bad')\n"
        "def bad(a: int) -> int:\n"
        "    '''x'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        "    return [Instance(args={'a': 1}) for _ in range(n)]\n"
        "def judge(instance, output):\n"
        "    return output == THRESHOLD\n"
    )
    (pa, da, ja), (pb, db, jb) = _layer_pair(tmp_path, shared, shared.replace("THRESHOLD = 1", "THRESHOLD = 2"))
    assert (pa, da) == (pb, db)  # 题型与分布不变
    assert ja != jb  # 仅判定段变


def test_layer_anchor_order_independent(tmp_path: Path) -> None:
    """归层与锚点源码顺序无关: gen/judge 定义在 @case fn 之前, judge 域常量仍归判定层
    (间接断言: 改 judge 域常量不动题型段 — 若顺序影响归层, THRESHOLD 会误归题型层)"""
    reordered = (
        "from ai_powered.llm_bench import Instance, case\n"
        "SPAN = 5\n"
        "THRESHOLD = 1\n"
        "def gen(rng, n):\n"
        "    return [Instance(args={'a': rng.randrange(SPAN)}) for _ in range(n)]\n"
        "def judge(instance, output):\n"
        "    return output == THRESHOLD\n"
        "@case(id='neg/bad')\n"
        "def bad(a: int) -> int:\n"
        "    '''x'''\n"
        "    ...\n"
    )
    b = _mini_suite(tmp_path, "b", case_body=reordered)
    c = _mini_suite(tmp_path, "c", case_body=reordered.replace("THRESHOLD = 1", "THRESHOLD = 2"))
    pb, db, jb = _layers(b)
    pc, dc, jc = _layers(c)
    assert pb == pc  # 题型不变 (THRESHOLD 仍归判定层)
    assert jb != jc  # 判定段变


def test_digest_rejects_multiple_case_defs_in_one_file(tmp_path: Path) -> None:
    """格式契约: 一文件一 @case — 第二个 @case 定义 fail-fast
    (loader 按属性取第一个, digest 若取别的会语义分叉; 入口拒绝优于半路失败)"""
    body = _GOOD_CASE + (
        "\n@case(id='neg/extra')\n"
        "def extra(a: int) -> int:\n"
        "    '''y'''\n"
        "    ...\n"
        "def _unused_gen(rng, n):\n"
        "    return []\n"
    )
    d = _mini_suite(tmp_path, case_body=body)
    with pytest.raises(ValueError, match="multiple @case"):
        apb.suite_digests(d)


def test_digest_rejects_multiple_case_defs_idless_bait(tmp_path: Path) -> None:
    """哨根因回归: 第一个 @case 合法 keyword-only 但无 id (digest 侧取不到 id) 不得放行第二个 —
    "已见过 @case" 与 "已提取合法 id" 是两个独立事实, 守卫 (而非 id 检查) 必须是此形态的唯一拒绝路径;
    若退化回 "case_id is not None" 判据, 本测试 DID NOT RAISE (digest-only 路径静默算出指纹)"""
    body = (
        "from ai_powered.llm_bench import Instance, case\n"
        "@case(difficulty='easy')\n"
        "def first(a: int) -> int:\n"
        "    '''x'''\n"
        "    ...\n"
        "@case(id='kw/id')\n"
        "def second(a: int) -> int:\n"
        "    '''y'''\n"
        "    ...\n"
        "def gen(rng, n):\n"
        "    return [Instance(args={'a': 1}, expected=1) for _ in range(n)]\n"
    )
    d = _mini_suite(tmp_path, case_body=body)
    with pytest.raises(ValueError, match="multiple @case"):
        apb.suite_digests(d)


def test_digest_rejects_positional_case_arguments(tmp_path: Path) -> None:
    """@case 是 keyword-only 签名: 位置传参在 loader 是 TypeError, digest-only 路径对齐拒绝"""
    body = _GOOD_CASE.replace("@case(id='neg/bad')", "@case('neg/bad')")
    d = _mini_suite(tmp_path, case_body=body)
    with pytest.raises(ValueError, match="keyword-only"):
        apb.suite_digests(d)


def test_suite_digests_rejects_empty_suite(tmp_path: Path) -> None:
    """空 suite (只有 suite.py) 在 digest-only 路径也 fail-fast — 防 builtin 盘点静默放行误清空目录"""
    d = tmp_path / "empty"
    d.mkdir()
    (d / "suite.py").write_text("from ai_powered.llm_bench import SuiteMeta\nSUITE_META = SuiteMeta(name='empty', rotation='day')\n")
    with pytest.raises(ValueError, match="no case files"):
        apb.suite_digests(d)


def test_digest_insensitive_to_comments_and_formatting(tmp_path: Path) -> None:
    noisy = _GOOD_CASE.replace("    ...\n", "    ...  # tail comment\n\n\n")
    a, b = _mini_suite(tmp_path, "a"), _mini_suite(tmp_path, "b", case_body=noisy)
    assert apb.suite_digest(a) == apb.suite_digest(b)


def test_digest_sensitive_to_docstring(tmp_path: Path) -> None:
    """docstring 是 prompt 的一部分, 改动 = 换题"""
    changed = _GOOD_CASE.replace("'''x'''", "'''y'''")
    a, b = _mini_suite(tmp_path, "a"), _mini_suite(tmp_path, "b", case_body=changed)
    assert apb.suite_digest(a) != apb.suite_digest(b)


def test_digest_rejects_duplicate_case_id(tmp_path: Path) -> None:
    """id 是哈希键: 同 suite 重复 id 静默折叠 = 身份谎报, fail-fast 拒绝"""
    d = _mini_suite(tmp_path)
    (d / "dup.py").write_text(_GOOD_CASE)  # 与 bad.py 同 id 'neg/bad'
    with pytest.raises(ValueError, match="duplicate case id"):
        apb.suite_digests(d)
    with pytest.raises(ValueError, match="duplicate case id"):
        load_suite(d)


def test_digest_rejects_non_literal_case_id(tmp_path: Path) -> None:
    """id 无法静态识别 (非字符串字面量) = 无法稳定进哈希键, fail-fast 拒绝 (AST 层, 不 import)"""
    body = _GOOD_CASE.replace("@case(id='neg/bad')", "@case(id=SOME_CONST)")
    d = _mini_suite(tmp_path, case_body=body)
    with pytest.raises(ValueError, match="string literal"):
        apb.suite_digests(d)


def test_digest_rejects_case_without_decorator(tmp_path: Path) -> None:
    body = _GOOD_CASE.replace("@case(id='neg/bad')\n", "")
    d = _mini_suite(tmp_path, case_body=body)
    with pytest.raises(ValueError, match="no @case"):
        apb.suite_digests(d)


def test_case_meta_field_coverage() -> None:
    """守护: CaseMeta 新增字段时须评估指纹归属 —— 元数据默认被剔除 (不进指纹);
    若新字段携带测量语义 (如计分权重), 须显式纳入指纹并在 bench design.md 记录"""
    from ai_powered.llm_bench import CaseMeta

    assert set(CaseMeta.__dataclass_fields__) == {"id", "difficulty", "source", "description"}


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


# --- python_expression judge 的 DoS 缓解与良性语义守护 (llm-bench#12 上游半) ---

#: 攻击样本 = 被测模型的原始响应 (judge 的直接输入): 资源耗尽族, 修复后须全拒 (judge → False)
_DOS_ATTACK_OUTPUTS = [
    "9**9**9**9",  # 右结合幂链: 求值构造天文数字大整数
    "2 ** " * 1500 + "2",  # 深链幂 (静态分析递归超栈须收敛为拒绝)
    "pow(9, 9 ** 9)",  # 调用面等价物 (pow 入白名单后与 ** 算子同水位防护)
    "'a' * (10 ** 9)",  # 长序列构造: 重复
    "[0] * (2 ** 30)",  # 长序列构造: 列表重复
    "([0] * 10 ** 5 + [0] * 10 ** 5) * 10 ** 5",  # 长序列构造: 拼接复合
]


def test_judge_python_expression_dos_outputs_rejected_bounded() -> None:
    """DoS 样本经 judge 全拒且有界时间内收敛 (llm-bench#12)。
    子进程 + 超时隔离: 防御失效 (回归) 时表现为本测试快速失败, 而非 judge 挂死/OOM 测试进程本身。
    甄别力双通道: 非超时样本以 "守卫明确拒绝 (SafeEvalViolation, 非求值错误被吞)" 失败;
    挂死型样本 (9**9**9**9 等) 由超时兜底 — 甄别不集中于单一样本"""
    child = (
        "from pathlib import Path\n"
        "import sys\n"
        "from ai_powered.llm_bench import Instance\n"
        "from ai_powered.llm_bench.cases import _import_module\n"
        "from ai_powered.utils.safe_eval import SafeEvalViolation, safe_eval\n"
        "mod = _import_module(Path(sys.argv[1]), 'pe_case')\n"
        "inst = Instance(args={'expr': '2^3'}, expected=8)\n"
        "for out in sys.argv[2:]:\n"
        "    try:\n"
        "        safe_eval(out)\n"
        "    except SafeEvalViolation:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'not rejected by guard: {out[:60]}')\n"
        "    assert not mod.judge(inst, out), f'dos output accepted: {out[:60]}'\n"
        "print('OK')\n"
    )
    # PYTHONPATH 从已导入包自身推导 (对齐 test_sandbox.py 先例): 防 worktree 场景 editable-install 指向主仓库旧代码
    env = {**os.environ, "PYTHONPATH": str(Path(apb.__file__).resolve().parents[2])}
    proc = subprocess.run(
        [sys.executable, "-c", child, str(_FUNC_SIMU / "python_expression.py"), *_DOS_ATTACK_OUTPUTS],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert proc.returncode == 0, f"dos sample escaped guard or crashed: rc={proc.returncode} stderr={proc.stderr}"
    assert proc.stdout.strip() == "OK"


def _canonical_translation(expr: str) -> str:
    """gen 表面文本的机械正确译法 (模拟满分模型): ^ → **, sum_from_to(a, b) → sum(range(a, b + 1))"""
    body = re.sub(r"sum_from_to\((\d+), (\d+)\)", r"sum(range(\1, \2 + 1))", expr)
    return body.replace("^", "**")


def test_judge_python_expression_benign_semantics_preserved() -> None:
    """良性判定语义不变 (llm-bench#12 DoD): gen 域全部实例的机械正确译法必须通过 ——
    DoS 加固只拒攻击样本, 不得误伤正确答案; 显式钉各题型的规范译法 (含 pow, bench 侧已 pin 的语义)"""
    case = _case("func-simu/python-expression")
    assert case.judge is not None
    for inst in case.gen(random.Random(20260826), 50):
        assert case.judge(inst, _canonical_translation(inst.args["expr"])), f"benign output rejected: {inst.args['expr']!r}"
    pinned = [
        ("2^3", "2 ** 3", 8),  # ^ → ** 直译
        ("2^3", "pow(2, 3)", 8),  # ^ → pow 内置函数译法
        ("sum_from_to(1, 4)", "sum(range(1, 5))", 10),  # 求和函数译法
        ("(5 + 3) * 2", "(5 + 3) * 2", 16),
        ("(3 - 20)", "(3 - 20)", -17),  # 负值真值
    ]
    for surface, output, expected in pinned:
        assert case.judge(Instance(args={"expr": surface}, expected=expected), output), output
        assert not case.judge(Instance(args={"expr": surface}, expected=expected), f"({output}) + 1")  # 值错拒


def test_judge_gen_user_info_structure() -> None:
    # UserInfo 定义在 suite 文件内, 按路径加载取类 (与 load_suite 同款加载方式)
    from ai_powered.llm_bench.cases import _import_module

    mod = _import_module(_FUNC_SIMU / "gen_user_info.py", "test_gen_user_info_mod")
    judge = mod.judge
    user_info_cls = mod.UserInfo
    assert judge(Instance(args={}), user_info_cls(name="Alice", country="China", age="30"))
    assert not judge(Instance(args={}), user_info_cls(name="  ", country=None, age=None))  # 空白名
    assert not judge(Instance(args={}), "not a userinfo")  # 类型不符
