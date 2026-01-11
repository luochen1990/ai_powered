import re
from typing import Optional

_json_pattern = re.compile(r'({.*})', re.DOTALL)


def extract_json_from_message(message: str) -> Optional[str]:
    """Extract JSON from markdown code blocks and plain JSON."""

    # Try to extract JSON from markdown code blocks first
    # Use [^] to exclude newlines from matching
    markdown_pattern = re.compile(r'```(?:json)?\s*(?:.*?)\s*```', re.DOTALL)
    match = markdown_pattern.search(message)
    if match:
        # Extract JSON and clean it
        return match.group(1)

    # Fallback to original method for non-markdown content
    reversed_message = message[::-1]

    # 在反转消息中寻找第一个完整的 {} 对 (在反转后是 }...{)
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

    return None
