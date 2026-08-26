# suite 内容指纹 — suite 身份的机械判定 (分层两级: 题型*分布:判定)
#
# 职责边界: suite 目录 → per-case 分层指纹 + suite 级分层指纹, 供 payload/榜单区分"跑了哪一版内容"
# 格式 (llm-bench#9 P3 定稿, 2026-08): 三段级联 "<prompt>*<distribution>:<judging>", 每段 16 hex;
#   前缀可比 —— 比较到不同切断点得到不同粒度:
#     段1 相等            ⟺ 同题型   (签名+docstring+类型依赖 — prompt 侧内容)
#     段1*段2 前缀相等     ⟺ 同题库   (再加实例分布 — gen 及其专属依赖)
#     全串相等            ⟺ 统计可比 (再加判定 — judge/expected 语义)
#   种子 (suite, period)→seed 不入指纹: 采样点身份与分布身份正交, 由 payload 字段承载
#   (day rotation 跨种子统计可比是榜单聚合的前提); 逐题对账用 (digest, period) 复算实例集
# 级联与归层 (防漏报的关键):
#   - 段间级联: 段2 哈希输入含段1, 段3 含段2 —— 后段身份以全部更粗段为前缀, 前缀比较语义成立;
#     归因看"第一个变化的段" (级联只向后方传导, 前段不变 ⟹ 前段语义内容未变)
#   - 顶层语句按"引用它的最粗锚点层"归层 (题型 < 分布 < 判定): 多层共用符号 (如 UserInfo 同时被
#     签名与 judge 引用) 归题型层, 其变更经级联传导至所有段 —— 任何粒度的比较都如实报告差异
#   - expected 构造住 gen 内: (args, expected) 联合分布即"题目分布", expected 公式变更 = 分布段变更
#   - 无人引用的顶层语句归题型层 (保守: 宁可误报新版本, 不漏报)
# 契约:
#   - @case 元数据 (id/difficulty/source/description) 剔除出指纹 — 标签层非测量内容;
#     改 id = 换题号: case 各段不变, suite 聚合键变 (suite 全串变)
#   - name/rotation 不进指纹 — 分布身份与采样点身份正交, 详见 llm-bench#9 决策记录
#   - 文件名解耦: 聚合按 case id 排序 (非文件名)
#   - id 唯一性 / 一文件一 @case (参数须 keyword-only) / id 须为字符串字面量 / 须有 @case 装饰器
#     / suite 非空 — fail-fast (digest-only 路径如 builtin 盘点与 loader 路径同等拒绝)
#   - 排除非语义噪声: 注释/空行/格式化差异经 parse→unparse 往返归一 (docstring 保留 — prompt 的一部分)
#   - 无密钥: 任何持有 suite 目录者可复算 (审计/服务端复核用; 区别于 integrity 的 HMAC 完整性封条)
#   - 跨解释器稳定性: ast.unparse (而非 ast.dump — 后者空字段打印随版本漂移), 用例语法子集 3.10+ 稳定,
#     由 pinned digest 金丝雀守护 —— 解释器升级引入 unparse 风格漂移时金丝雀变红 = "身份静默断代"报警
#   - 改哈希输入集合 (段结构/归层规则/哈希对象增删) = 身份断代, 须在 bench 仓 design.md 记录;
#     CaseMeta 新增字段默认不进指纹, 测量语义字段 (如计分权重) 须显式纳入并记录
# 迁移: 自 llm-bench src/llm_bench/suite_digest.py 迁入后分层化 (ai-powered#12 / llm-bench#9 P3);
#   bench 侧 re-export, 消费点 (payload/金丝雀/submit 校验) 在 bench 仓

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeGuard

from .canonical import canonical_json

DIGEST_LEN = 16  # 每段长度 (hex)

_LAYER_PROMPT = 0
_LAYER_DISTRIBUTION = 1
_LAYER_JUDGING = 2
_N_LAYERS = 3


@dataclass(frozen=True)
class LayeredDigest:
    """分层指纹: 题型/分布/判定三段 (各 16 hex, 段间级联构造)

    比较语义只有前缀 (后段身份以更粗段为前缀, 单独比较后段无意义);
    str() 合成串按 "<prompt>*<distribution>:<judging>" 格式, 前缀即切断点"""

    prompt: str
    distribution: str
    judging: str

    def __str__(self) -> str:
        return f"{self.prompt}*{self.distribution}:{self.judging}"


def case_files(suite_dir: Path) -> list[Path]:
    """参与加载的 case 文件 (发现规则 SSOT: *.py 排除 _ 前缀与 suite.py, 按文件名排序);
    load_suite 与 digest 共用 — 保证 "digest 覆盖集 == 实际加载集" 永不漂移"""
    return [p for p in sorted(suite_dir.glob("*.py")) if not p.name.startswith("_") and p.name != "suite.py"]


def _defined_names(stmt: ast.stmt) -> set[str]:
    """顶层语句定义的名字: 函数/类名, 赋值目标 (含元组解包), AnnAssign 目标, import 别名。

    未覆盖形态 (for/walrus/AugAssign/嵌套定义等) 的名字不进索引 → 其定义语句必落题型层
    (保守方向: 宁可误报新版本, 不漏报), 属设计意图而非疏漏"""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    if isinstance(stmt, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in stmt.names}
    if isinstance(stmt, ast.ImportFrom):
        return {a.asname or a.name for a in stmt.names}
    return set()


def _referenced_names(stmt: ast.stmt) -> set[str]:
    """语句引用的名字 (含注解/装饰器/嵌套; 粗粒度 — 只需覆盖顶层符号依赖)"""
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}


def _is_case_call(dec: ast.expr) -> TypeGuard[ast.Call]:
    """@case 装饰器识别 (按 AST 名字 — 别名引用不识别, 格式契约: case 须原名引用)"""
    return isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "case"


def _case_id_and_buckets(path: Path) -> tuple[str, list[list[str]]]:
    """case 文件 → (case_id, 三层规范化语句列表): AST 一次解析多用途

    id 从 @case(id=...) 字面量提取, 装饰器整体剔除 (元数据出指纹);
    三分区归层: 锚点 (@case fn→题型, gen→分布, judge→判定), 其余顶层语句按
    "引用它的最粗锚点层"归层 (传递闭包), 无人引用 → 题型层 (保守)
    格式契约: 一文件一 @case (多个 fail-fast — loader 按属性取第一个, 与 digest 的
    语义在此分叉, 与其半路失败不如入口拒绝)"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = list(tree.body)
    case_id: str | None = None
    saw_case = False  # "已见过 @case" 与 "已提取合法 id" 是两件事 — 位置传 id 的 @case 只置前者
    anchor_layers: dict[int, int] = {}

    for i, stmt in enumerate(top):
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        dec = next((d for d in stmt.decorator_list if _is_case_call(d)), None)
        if dec is not None:
            if saw_case:
                raise ValueError(f"{path}: multiple @case definitions in one file (one case per file)")
            saw_case = True
            if dec.args:  # @case 是 keyword-only 签名; 位置传参在 loader 是 TypeError, digest 侧对齐拒绝
                raise ValueError(f"{path}: @case arguments must be keyword-only (got positional: {ast.unparse(dec.args[0])})")
            stmt.decorator_list.remove(dec)  # 元数据出指纹 (ast 节点恒等, remove 即剔这一个)
            anchor_layers[i] = _LAYER_PROMPT
            for kw in dec.keywords:
                if kw.arg == "id":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        case_id = kw.value.value
                    else:
                        raise ValueError(f"{path}: @case id must be a string literal (got {ast.unparse(kw.value)})")
                    break
        elif isinstance(stmt, ast.FunctionDef) and stmt.name in ("gen", "judge"):
            anchor_layers[i] = _LAYER_DISTRIBUTION if stmt.name == "gen" else _LAYER_JUDGING
    if case_id is None:
        raise ValueError(f"{path}: no @case(id=...) decorated definition found")

    # 顶层符号依赖 → 各锚点沿引用名传播"需求层" (传递闭包, 多锚点命中取最粗)
    definer = {n: i for i, s in enumerate(top) for n in _defined_names(s)}
    layer_of: dict[int, int] = dict(anchor_layers)
    for anchor, layer in anchor_layers.items():
        seen, stack = {anchor}, [anchor]
        while stack:
            j = stack.pop()
            layer_of[j] = min(layer_of.get(j, _N_LAYERS), layer)
            for k in {definer[n] for n in _referenced_names(top[j]) if n in definer}:
                if k not in seen:
                    seen.add(k)
                    stack.append(k)

    buckets: list[list[str]] = [[] for _ in range(_N_LAYERS)]
    for i, stmt in enumerate(top):
        buckets[layer_of.get(i, _LAYER_PROMPT)].append(ast.unparse(stmt))
    return case_id, buckets


def _hash_layer(prev: str, content: object) -> str:
    """段哈希 (级联): SHA256(前段 digest ‖ canonical_json(本层内容))[:16]"""
    return hashlib.sha256((prev + canonical_json(content)).encode("utf-8")).hexdigest()[:DIGEST_LEN]


def _case_layered_digest(path: Path) -> tuple[str, LayeredDigest]:
    """单 case 文件 → (case_id, 分层指纹): 三段级联 (题型 → 分布 → 判定)"""
    case_id, buckets = _case_id_and_buckets(path)
    prompt = _hash_layer("", buckets[_LAYER_PROMPT])
    distribution = _hash_layer(prompt, buckets[_LAYER_DISTRIBUTION])
    judging = _hash_layer(distribution, buckets[_LAYER_JUDGING])
    return case_id, LayeredDigest(prompt, distribution, judging)


def suite_digests(suite_dir: Path) -> dict[str, LayeredDigest]:
    """suite 目录 → {case_id: 分层指纹}; 重复 id / 空 suite 均 fail-fast
    (id 是聚合键静默折叠 = 身份谎报; 空 suite 出合法指纹 = 盘点口径静默放行误清空目录)"""
    result: dict[str, LayeredDigest] = {}
    for path in case_files(suite_dir):
        case_id, layered = _case_layered_digest(path)
        if case_id in result:
            raise ValueError(f"duplicate case id {case_id!r} in suite {suite_dir} (from {path.name})")
        result[case_id] = layered
    if not result:
        raise ValueError(f"no case files found in {suite_dir}")
    return result


def suite_digest_from_cases(cases: Mapping[str, LayeredDigest]) -> str:
    """suite 级分层指纹 (级联聚合, 按 case id 排序): 段内聚合 per-case 同层段, 段间级联;
    输出与 per-case 指纹同格式 (str(LayeredDigest) 单点渲染)"""
    ids = sorted(cases)
    s1 = _hash_layer("", [[i, cases[i].prompt] for i in ids])
    s2 = _hash_layer(s1, [[i, cases[i].distribution] for i in ids])
    s3 = _hash_layer(s2, [[i, cases[i].judging] for i in ids])
    return str(LayeredDigest(s1, s2, s3))


def suite_digest(suite_dir: Path) -> str:
    """suite 目录的内容指纹 (便捷入口): 聚合 suite_digests 的分层指纹"""
    return suite_digest_from_cases(suite_digests(suite_dir))
