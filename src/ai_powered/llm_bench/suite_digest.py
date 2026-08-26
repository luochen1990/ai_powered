# suite 内容指纹 — suite 身份的机械判定 (内容哈希)
#
# 职责边界: suite 目录 → 16 hex digest, 供 payload/榜单区分"跑的是哪一版内容"
# 契约:
#   - digest 相同 ⟺ 测试内容等价: 各参与加载的 case 文件 (排除 _ 前缀与 suite.py) 的规范化源码
#     + suite 的 name/rotation (两者参与种子派生与周期语义, description 纯展示不进)
#   - 排除非语义噪声: 注释/空行/格式化差异经 parse→unparse 往返归一 (docstring 保留 — 它是 prompt 的一部分)
#   - 无密钥: 任何持有 suite 目录者可复算 (审计/服务端复核用; 区别于 integrity 的 HMAC 完整性封条)
#   - 跨解释器稳定性: 规范化用 ast.unparse (而非 ast.dump — 后者对空字段的打印随版本漂移, 3.12/3.14 实测不同);
#     对用例语法子集 (函数/类/常量/装饰器/注解) 在 3.10+ 稳定, 由 test/unit/test_llm_bench_suite.py 的
#     pinned digest 金丝雀守护 —— 解释器升级引入 unparse 风格漂移时金丝雀变红, 即"身份静默断代"的报警器
#   - 改哈希输入集合 (增删哈希对象) = suite 身份断代, 须记录; SuiteMeta 增字段同理 (有守护测试)
# 已知取舍: case 文件按文件名排序进列表 — 改文件名会改变 digest (过度敏感方向安全: 宁可误报新版本, 不漏报)
# 迁移: 自 llm-bench src/llm_bench/suite_digest.py 迁入 (ai-powered#12 / llm-bench#9 P1);
#   digest 两级化 (llm-bench#9 P3) 将在本文件实施, bench 侧 re-export

import ast
import hashlib
from pathlib import Path

from .canonical import canonical_json

DIGEST_LEN = 16


def case_files(suite_dir: Path) -> list[Path]:
    """参与加载的 case 文件 (发现规则 SSOT: *.py 排除 _ 前缀与 suite.py, 按文件名排序);
    load_suite 与 digest 共用 — 保证 "digest 覆盖集 == 实际加载集" 永不漂移"""
    return [p for p in sorted(suite_dir.glob("*.py")) if not p.name.startswith("_") and p.name != "suite.py"]


def _module_canonical_source(path: Path) -> str:
    """单文件的规范化表示: parse → unparse 往返 (丢弃注释/空白差异, 保留全部语义与 docstring)"""
    return ast.unparse(ast.parse(path.read_text(encoding="utf-8")))


def suite_digest(suite_dir: Path, name: str, rotation: str) -> str:
    """suite 目录的内容指纹: SHA256(canonical_json({name, rotation, case 规范化源码}))[:16]"""
    case_sources = [_module_canonical_source(p) for p in case_files(suite_dir)]
    payload = canonical_json({"name": name, "rotation": rotation, "cases": case_sources})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_LEN]
