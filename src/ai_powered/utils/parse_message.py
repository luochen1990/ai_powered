import re
from typing import Optional

_json_pattern = re.compile(r'({.*})', re.DOTALL)


def extract_json_from_message(message: str) -> Optional[str]:
    reversed_message = message[::-1]

    # 在反转消息中寻找第一个完整的 {} 对
    match = _json_pattern.search(reversed_message)
    if match:
        # 获取反转字符串中匹配的起始和结束索引
        start, end = match.span()

        # 计算原始字符串中的索引
        original_start = len(message) - end
        original_end = len(message) - start

        # 从原始消息中提取 JSON 字符串
        json_str = message[original_start:original_end]

        return json_str
    else:
        return None
