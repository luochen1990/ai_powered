# ai_powered.llm_bench — llm-bench 基准的契约与套件承载包

本目录是 llm-bench 基准的格式契约与 suite 内容 SSOT（方案: llm-bench#9，实施: ai-powered#12）。
llm-bench 仓单向 runtime 依赖本包，其 `cases.py` 薄化为 re-export；依赖方向严格 **llm-bench → ai-powered**，
本包内禁止 import llm_bench。

## 结构

- `cases.py` — 格式契约（`@case` / `Instance` / `SuiteMeta` / `CaseMeta` / MISSING 哨兵）+ suite 发现/加载/加载时校验。加载器只此一份
- `suite_digest.py` + `canonical.py` — suite 内容指纹（**分层两级**：per-case 三段 `题型*分布:判定` + 按 case id 聚合的 suite 合串；**前缀比较 = 粒度选择**：段1 ⟺ 同题型、段1\*段2 ⟺ 同题库、全串 ⟺ 统计可比）。级联构造、归层规则、剔除项（种子/name/rotation/@case 元数据）等契约见 `suite_digest.py` 头注（SSOT）。`case_files()` 发现规则与 `load_suite` 共用，保证 "digest 覆盖集 == 实际加载集" 永不漂移；case id 从 `@case(id=...)` 字面量提取（一文件一 @case、非字面量 id、重复 id、空 suite 均 fail-fast）
- `suites/` — 纯数据目录（**非包**，无 `__init__.py`，运行时按路径逐文件加载）。当前含 `func-simu/`（溯源自 `test/examples/ai_powered_decorator@a1a39e0`；相对上游的有意偏离已在各文件头注标注）

## 边界纪律

1. **`suites/` 内任何变更 = bench suite 身份断代**（digest 变，`test/unit/test_llm_bench_suite.py` 的 pinned digest 金丝雀会红）：须与 bench 侧协调 bump 依赖 rev 后同步重 pin，**不允许随手改**
2. **不是所有属性测试都进 bench**：非 bench 用途的测试住别处（如 `test/`），不进 `suites/`
3. **`cases.py` 的 `@case`/`Instance`/`SuiteMeta`/`CaseMeta`/`load_suite`（及 `canonical_json` 等 digest/integrity 原语）是对外公共 API**（llm-bench 依赖它），变更需两仓协调

## 静态门禁说明

`suites/` 不走 pyright 静态门禁（pyproject 中 ignore）——suite 内容是用例数据而非库代码，合法性由加载时校验（`_validate_case`）+ 单测 + e2e adapter 测试守护。
