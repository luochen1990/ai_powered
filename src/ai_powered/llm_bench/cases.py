# 用例格式与套件加载 — bench 数据模型的 SSOT (格式契约 + 发现/加载/校验)
#
# 职责边界: @case 装饰器 (任务声明) / Instance (测试实例) / suite 目录发现与加载 + 加载时校验
# 用例 = 任务声明 (签名+docstring, 进 prompt) + 实例生成器 gen (种子驱动) + 判定方式 (三形态):
#   1. 等值:  Instance.expected 与模型输出结构相等 (未定义 judge 时默认)
#   2. 可执行: 模块级 judge(instance, output) -> bool, 执行输出并比对 (如 safe_eval)
#   3. 属性:  judge 校验结构性质 (生成型任务无唯一真值)
# 契约:
#   - 用例文件禁止 `from __future__ import annotations` (签名注解须为真实类型对象, msgspec 依赖)
#   - gen 必须只用传入 rng 取随机数 (确定性); expected 通过返回类型校验 (msgspec 往返)
#   - expected 可为 None (Optional 语义); "无期望" 用 MISSING 哨兵表示且必须配 judge
# 迁移 (ai-powered#12 / llm-bench#9 P1): 自 llm-bench src/llm_bench/cases.py 迁入,
#   bench 侧薄化为 re-export; 本包公共 API (@case/Instance/SuiteMeta/CaseMeta/load_suite) 变更需两仓协调。
#   迁移时的显式行为决策 (见 ai-powered#12 定稿评论):
#   - 探针种子改用裸 random.Random(probe_seed) (原 llm_bench.seeding.case_rng 为反向依赖, R2)
#   - 动态加载的 sys.modules 键前缀改为 ai_powered_llm_bench_* (避免与 bench 侧加载器键冲突)

import importlib.util
import inspect
import random
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, cast

import msgspec

from .suite_digest import case_files, suite_digest

# "无期望值" 哨兵: 与 None (合法期望值) 区分
MISSING: Any = object()


@dataclass(frozen=True)
class SuiteMeta:
    """suite.py 中 SUITE_META 的类型 (套件级元数据)"""

    name: str
    rotation: Literal["day", "week"]
    description: str = ""


@dataclass(frozen=True)
class CaseMeta:
    """@case 装饰器携带的元数据"""

    id: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source: str = ""  # 溯源标注, 如 "ai_powered:test/examples/.../add.py@a1a39e0"
    description: str = ""


@dataclass
class Instance:
    """单个测试实例: 模型输入参数 + 期望 (等值判定用)"""

    args: dict[str, Any]
    expected: Any = MISSING


# 模块级 judge 签名: (instance, output) -> 是否通过
JudgeFn = Callable[[Instance, Any], bool]
# 模块级 gen 签名: (rng, n) -> 实例列表
GenFn = Callable[[random.Random, int], list[Instance]]


@dataclass
class Case:
    """加载完成的用例 (声明 + 生成器 + 判定器)"""

    meta: CaseMeta
    fn: Callable[..., Any]  # 任务声明函数 (函数体恒为 ..., 永不本地执行)
    gen: GenFn
    judge: JudgeFn | None


@dataclass
class Suite:
    """加载完成的套件"""

    name: str
    rotation: Literal["day", "week"]
    description: str
    digest: str  # 内容指纹 (suite_digest.py), suite 身份的机械判定
    cases: list[Case]  # 按 case id 排序


def case(
    *,
    id: str,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    source: str = "",
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """声明一个 bench 用例 (装饰任务声明函数)"""
    meta = CaseMeta(id=id, difficulty=difficulty, source=source, description=description)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._llm_bench_case = meta  # type: ignore[attr-defined]
        return fn

    return decorator


def _import_module(path: Path, module_name: str) -> ModuleType:
    """按文件路径加载模块 (suite 目录不是包, 逐文件独立加载)

    module_name 仅作 sys.modules 键 (spec_from_file_location 接受任意字符串),
    不要求合法 Python 标识符 — suite 目录名可含连字符 (如 func-simu)
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass 等机制依赖 module 注册
    spec.loader.exec_module(module)
    return module


def _load_suite_meta(suite_dir: Path) -> Any:
    """加载 suite.py (SUITE_META: name/rotation/description)"""
    meta_path = suite_dir / "suite.py"
    if not meta_path.exists():
        raise ValueError(f"suite.py not found in {suite_dir}")
    module = _import_module(meta_path, f"ai_powered_llm_bench_suite_{suite_dir.name}")
    meta = getattr(module, "SUITE_META", None)
    if meta is None:
        raise ValueError(f"SUITE_META not defined in {meta_path}")
    return meta


def _validate_case(case: Case, probe_seed: int) -> None:
    """加载时校验 (fail-fast): 注解完备 / args 可绑定 / expected 类型正确 / judge 契约"""
    sig = inspect.signature(case.fn)

    for param in sig.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"[{case.meta.id}] parameter '{param.name}' lacks type annotation")
        if isinstance(param.annotation, str):
            raise ValueError(f"[{case.meta.id}] string annotation detected (from __future__ annotations?)")
    if sig.return_annotation is inspect.Parameter.empty:
        raise ValueError(f"[{case.meta.id}] return annotation is required")

    # 探针实例: 用固定探针种子生成, 校验生成器行为 (不进入真实跑分)
    # (迁移决策 R2: 探针只需确定性, 裸 Random 即可, 不引入对 bench 种子机制的依赖)
    probe_instances = case.gen(random.Random(probe_seed), 4)
    if not isinstance(probe_instances, list) or len(probe_instances) < 1:  # pyright: ignore[reportUnnecessaryIsInstance] — 插件契约的运行时防御
        raise ValueError(f"[{case.meta.id}] gen returned no instances")

    return_type = sig.return_annotation
    for i, inst in enumerate(probe_instances):
        try:
            sig.bind(**inst.args)
        except TypeError as e:
            raise ValueError(f"[{case.meta.id}] instance {i} args do not bind to signature: {e}") from e
        if case.judge is None:
            if inst.expected is MISSING:
                raise ValueError(f"[{case.meta.id}] instance {i} has no expected and case defines no judge")
            # expected 必须能通过返回类型 (msgspec 编解码往返)
            try:
                msgspec.json.decode(msgspec.json.encode(inst.expected), type=return_type)
            except Exception as e:
                raise ValueError(f"[{case.meta.id}] instance {i} expected does not conform to return type: {e}") from e


@cache
def builtin_suite_digests() -> dict[str, str]:
    """包内自带 suite 名 → 当前内容指纹 (服务端盘点标记"现行内容"用; 只算指纹不做用例加载校验)
    suite 随包分发、运行期不可变, 故缓存; 目录内无 suite.py 的子目录跳过
    只读契约: 返回值被进程级缓存, 调用方不得原地修改"""
    root = Path(__file__).parent / "suites"
    result: dict[str, str] = {}
    for suite_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (suite_dir / "suite.py").exists():
            continue
        meta = _load_suite_meta(suite_dir)
        result[meta.name] = suite_digest(suite_dir, meta.name, meta.rotation)
    return result


def load_suite(suite_dir: Path) -> Suite:
    """从目录加载套件: suite.py (元数据) + 其余每 *.py 一个用例 (加载 + 校验)"""
    suite_dir = suite_dir.resolve()
    meta = _load_suite_meta(suite_dir)
    if meta.name != suite_dir.name:
        raise ValueError(f"suite meta name '{meta.name}' != directory name '{suite_dir.name}'")

    # 探针种子: 仅用于加载校验, 与任何真实跑分种子空间无碰撞语义 (bench 侧种子机制见 llm_bench.seeding)
    probe_seed = -1

    cases: list[Case] = []
    for path in case_files(suite_dir):
        module = _import_module(path, f"ai_powered_llm_bench_case_{suite_dir.name}_{path.stem}")

        module_attrs: dict[str, Any] = vars(module)
        case_fn = next(
            (v for v in module_attrs.values() if callable(v) and hasattr(v, "_llm_bench_case")),
            None,
        )
        if case_fn is None:
            raise ValueError(f"no @case-decorated function in {path}")
        case_meta: CaseMeta = case_fn._llm_bench_case  # type: ignore[attr-defined]

        gen_attr = getattr(module, "gen", None)
        if not callable(gen_attr):
            raise ValueError(f"'gen(rng, n) -> list[Instance]' not defined in {path}")
        gen_fn: GenFn = cast(GenFn, gen_attr)
        judge_attr = getattr(module, "judge", None)
        judge_fn: JudgeFn | None = cast(JudgeFn, judge_attr) if callable(judge_attr) else None

        loaded = Case(meta=case_meta, fn=case_fn, gen=gen_fn, judge=judge_fn)
        _validate_case(loaded, probe_seed)
        cases.append(loaded)

    if not cases:
        raise ValueError(f"no cases found in {suite_dir}")
    cases.sort(key=lambda c: c.meta.id)

    digest = suite_digest(suite_dir, meta.name, meta.rotation)
    return Suite(name=meta.name, rotation=meta.rotation, description=meta.description, digest=digest, cases=cases)
