# canonical JSON — 确定性序列化基元
#
# 职责边界: 为 HMAC integrity / suite digest 提供"同一对象永远得到同一字符串"的序列化;
#           不做数据变换, 不处理 schema (那是 msgspec 的事)
#
# 规则 (RFC 8785 JCS 的子集, 同语言内完全确定):
#   - 键按字典序排序; 紧凑分隔符 (无空白); 非 ASCII 原样输出 (UTF-8)
#   - 禁止 NaN/Infinity (allow_nan=False, 出现即错误 —— 载荷中不应有)
#   - float 走 Python repr (3.1+ 最短往返表示, 跨版本稳定)
# 已知局限: 与完整 JCS 的差异在指数格式边角 (如 1e+21 vs 1e21), 仅影响跨语言校验, 当前不做
# 迁移: 自 llm-bench src/llm_bench/canonical.py 原样迁入 (ai-powered#12 / llm-bench#9 P1)

import json


def canonical_json(obj: object) -> str:
    """将可 JSON 化对象序列化为确定性字符串 (HMAC / digest 的输入基元)"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
