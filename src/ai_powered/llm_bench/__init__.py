# ai_powered.llm_bench — llm-bench 基准的格式契约与 suite 内容 SSOT
#
# 职责边界: bench 用例格式契约 (@case/Instance/SuiteMeta) + suite 发现/加载/校验 + 内容指纹 (digest)。
#   suite 纯数据住 suites/ (非包); 边界纪律与结构说明见本目录 AGENTS.md。
# 依赖方向: llm-bench 仓单向 runtime 依赖本包 (bench 侧 cases.py 薄化为 re-export);
#   本包是两仓公共 API, 变更需两仓协调 (方案: llm-bench#9, 实施: ai-powered#12)。

# bench 侧 P2/P3 (integrity HMAC / digest 两级化) 将经本包 re-export 此原语
from .canonical import canonical_json as canonical_json
from .cases import Case as Case
from .cases import CaseMeta as CaseMeta
from .cases import GenFn as GenFn
from .cases import Instance as Instance
from .cases import JudgeFn as JudgeFn
from .cases import MISSING as MISSING
from .cases import Suite as Suite
from .cases import SuiteMeta as SuiteMeta
from .cases import builtin_suite_digests as builtin_suite_digests
from .cases import case as case
from .cases import load_suite as load_suite
from .suite_digest import DIGEST_LEN as DIGEST_LEN
from .suite_digest import case_files as case_files
from .suite_digest import suite_digest as suite_digest
