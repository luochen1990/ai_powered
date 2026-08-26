# func-simu 属性 e2e — 包内 bench suite 的直接消费者 (ai-powered#12 / llm-bench#9 P1)
#
# 职责边界: 加载包内 func-simu suite → @ai_powered 动态包装 case.fn (真实 LLM 调用)
#           → gen 生成实例 (任意确定性 rng, 无需 bench 引擎/种子机制) → judge / 等值判定
# 定位: 本测试取代 examples "弱冒烟" 的深度盲区 (每 case 多实例 × 判定器);
#       test/examples/ai_powered_decorator 保持不动 (基础冒烟职责, 与 suite 内容无同步义务)
#
# 门禁语义 (bench 测量工具 ≠ CI 门禁的语义错配处理):
#   - bench suite 的 case 难度本就是为区分模型能力设计的, "模型必过"不是其契约;
#   - 系统性能力缺口 (每轮稳定复现) 以 xfail 显式豁免, 缺口消失自动转绿, 模型升级后靠评审清理;
#   - 偶发波动 (实例级 flaky) 沿用 CI 既有 --reruns 2 容忍 (与 examples 同款语义)

import random
from pathlib import Path
from typing import Any

import pytest

import ai_powered.llm_bench as apb
from ai_powered import ai_powered
from ai_powered.llm_bench import Case, Instance, load_suite

# 实例数: 与加载探针一致 (7 case × 4 = 28 次调用, 与现有 examples 测试量级相当)
_INSTANCES_PER_CASE = 4
# 任意确定性种子: 本仓 e2e 不依赖 bench 种子机制 (周期种子/防背题是 bench 侧语义)
_SEED = 42

# 已知模型能力缺口 (CI 基线模型 nemotron-120b:free, 选型史见 .forgejo/workflows/ci.yml 头注):
# - func-simu/gen-json-data: 深度遵循缺口 (豁免时曾 3 轮 12/12 实例复现输出空数组 [];
#   单实例偶发达标由 strict=False 容忍, 判定与归因保留 — 缺口修复即整体转绿)
_KNOWN_MODEL_GAPS = {"func-simu/gen-json-data"}


def _gap_reason(case_id: str) -> str:
    return f"CI 基线模型系统性能力缺口 [{case_id}], 见文件头注 _KNOWN_MODEL_GAPS"


_SUITE_DIR = Path(apb.__file__).parent / "suites" / "func-simu"
_SUITE = load_suite(_SUITE_DIR)
_REGISTRY: dict[str, tuple[Case, list[Instance], Any]] = {
    c.meta.id: (c, c.gen(random.Random(_SEED), _INSTANCES_PER_CASE), ai_powered(c.fn)) for c in _SUITE.cases
}

# 豁免清单一致性: gap 集合里的 id 必须真实存在 (suite 变更重命名/移除后豁免变 dead entry 会静默失效)
assert _KNOWN_MODEL_GAPS <= _REGISTRY.keys(), f"stale gap entries: {_KNOWN_MODEL_GAPS - _REGISTRY.keys()}"


def _marks(case_id: str) -> list[pytest.MarkDecorator]:
    if case_id in _KNOWN_MODEL_GAPS:
        return [pytest.mark.xfail(reason=_gap_reason(case_id), strict=False)]
    return []


# idx 上界从 gen 实际返回数量派生 (加载校验只保证 >=1, 不保证恰好 n)
_PARAMS: list[Any] = [
    pytest.param(case_id, idx, marks=_marks(case_id))
    for case_id, (_, instances, _) in sorted(_REGISTRY.items())
    for idx in range(len(instances))
]


@pytest.mark.parametrize(("case_id", "idx"), _PARAMS)
def test_func_simu_instance(case_id: str, idx: int) -> None:
    case, instances, wrapped = _REGISTRY[case_id]
    inst = instances[idx]
    output = wrapped(**inst.args)
    if case.judge is not None:
        assert case.judge(inst, output), f"judge failed [{case_id} #{idx}]: args={inst.args!r} output={output!r}"
    else:
        assert output == inst.expected, f"equality failed [{case_id} #{idx}]: args={inst.args!r} expected={inst.expected!r} output={output!r}"
