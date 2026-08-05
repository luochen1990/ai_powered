'''
职责: 验证本地 ollama 模型是否适合作为 ai_powered 的测试基准模型.

这是一份**模型选型/回归验证工具**, 不是功能测试. 与 test/ 目录的功能测试关系如下:

  - test/examples/        验证 ai_powered 代码功能 (装饰器/schema/解析), 经过项目代码全链路
  - scripts/this_file    验证模型本身的契约 (tool_calls/json_schema 是否可靠), 绕开项目代码
                          是 test/examples/ 的前置门禁: 先确认模型可靠, 再跑功能测试

当需要切换测试基准模型时, 用它快速确认候选模型能覆盖项目的三条核心调用路径, 避免选用
"工具调用不可靠"的模型导致 test/examples/ 测试套件随机 fail.

覆盖三条核心路径 (对齐 src/ai_powered/llm/adapters/*.py):
  1. tools 路径        — 模型应在 tool_calls 里返回结果 (ToolsFunctionSimulator)
  2. structured output — 模型应按 response_format=json_schema 返回 (StructuredOutputFunctionSimulator)
  3. chat fallback     — 模型应在 content 里返回 JSON (ChatFunctionSimulator)

每条路径重复 N 次, 统计成功率与时延, 并做一次复现性验证 (temperature=0 下输出应一致).

使用方式 (详见 justfile 的 test-model 入口):
  just test-model                    # 用 ollama.private.envrc 配置的默认模型
  just test-model --model llama3.1:8b  # 测其他候选模型

判定标准 (作为测试基准的基线, 满足度 ~70):
  - 三条路径成功率均 >= 80%
  - 复现性: 相同输入下输出一致

历史选型记录:
  - qwen3.6:27b (2026-08 实测通过, 详见 ollama.private.envrc 注释)
'''
import argparse
import json
import os
import re
import sys

import requests

# 从项目源码提取的真实 schema (见 test/examples/ai_powered_decorator/add.py + extract_user_info.py)
# 对应 ToolsFunctionSimulator/StructuredOutputFunctionSimulator 使用的 schema 形态
ADD_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_response",
        "schema": {
            "type": "object",
            "properties": {"result": {"type": "integer"}},
            "required": ["result"],
        },
    },
}

# tools 参数 (对齐 src/ai_powered/llm/adapters/tools_adapter.py:_param_tools_maker)
ADD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "return_result",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "integer"}},
                "required": ["result"],
            },
        },
    }
]

ADD_TOOL_CHOICE = {"type": "function", "function": {"name": "return_result"}}

# thinking 模式会消耗大量 token, 需留足余量, 否则模型还在思考就被 max_tokens 截断
MAX_TOKENS = 4000


def make_request(payload: dict) -> tuple[dict, str | None]:
    '''发起一次请求, 返回 (response_json, 错误信息)'''
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=180)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return {}, str(e)


def parse_json_object(text: str) -> dict | None:
    '''从可能包含 markdown 围栏或多余文本的内容中提取 JSON 对象'''
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def check_tools_path(model: str, prompt: str, expected: dict) -> tuple[bool, str, float]:
    '''路径1: tools 路径 — 期望模型在 tool_calls[0].function.arguments 返回 JSON'''
    import time
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You must call the return_result function."},
            {"role": "user", "content": prompt},
        ],
        "tools": ADD_TOOLS,
        "tool_choice": ADD_TOOL_CHOICE,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    resp, err = make_request(payload)
    elapsed = time.time() - t0
    if err:
        return False, f"请求失败: {err}", elapsed

    msg = resp.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    reasoning = msg.get("reasoning") or ""

    if not tool_calls:
        return False, f"未返回 tool_calls. content={msg.get('content','')[:80]!r} reasoning={reasoning[:80]!r}", elapsed

    args_str = tool_calls[0].get("function", {}).get("arguments", "")
    args = parse_json_object(args_str)
    if args is None:
        return False, f"tool_call.arguments 不是合法 JSON: {args_str[:80]!r}", elapsed
    if args.get("result") != expected["result"]:
        return False, f"result 错误: 期望 {expected['result']}, 实际 {args.get('result')}", elapsed

    tag = " [含thinking]" if reasoning else ""
    return True, f"OK ({elapsed:.1f}s){tag}", elapsed


def check_structured_output_path(model: str, prompt: str, expected: dict) -> tuple[bool, str, float]:
    '''路径2: structured output — 期望模型按 response_format=json_schema 返回 content'''
    import time
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return a JSON object matching the schema."},
            {"role": "user", "content": prompt},
        ],
        "response_format": ADD_SCHEMA,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    resp, err = make_request(payload)
    elapsed = time.time() - t0
    if err:
        return False, f"请求失败: {err}", elapsed

    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    if not content:
        return False, f"content 为空 (可能被 thinking 耗尽 max_tokens). reasoning_len={len(reasoning)}", elapsed

    parsed = parse_json_object(content)
    if parsed is None:
        return False, f"content 不是合法 JSON: {content[:80]!r}", elapsed
    if parsed.get("result") != expected["result"]:
        return False, f"result 错误: 期望 {expected['result']}, 实际 {parsed.get('result')}", elapsed

    tag = " [含thinking]" if reasoning else ""
    return True, f"OK ({elapsed:.1f}s){tag}", elapsed


def check_chat_fallback_path(model: str, prompt: str, expected: dict) -> tuple[bool, str, float]:
    '''路径3: chat fallback — 不给 tools, 只在 system 里说明 schema, 期望 content 是 JSON'''
    import time
    system_prompt = (
        "You must respond with a JSON object matching this schema: "
        '{"type":"object","properties":{"result":{"type":"integer"}},"required":["result"]}. '
        "Respond with ONLY the JSON, no other text."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    resp, err = make_request(payload)
    elapsed = time.time() - t0
    if err:
        return False, f"请求失败: {err}", elapsed

    msg = resp.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    if not content:
        return False, f"content 为空. reasoning={reasoning[:80]!r}", elapsed

    parsed = parse_json_object(content)
    if parsed is None:
        return False, f"content 中找不到 JSON 对象: {content[:80]!r}", elapsed
    if parsed.get("result") != expected["result"]:
        return False, f"result 错误: 期望 {expected['result']}, 实际 {parsed.get('result')}", elapsed

    tag = " [含thinking]" if reasoning else ""
    return True, f"OK ({elapsed:.1f}s){tag}", elapsed


def run_path(name: str, fn, model: str, runs: int) -> tuple[int, list[float]]:
    '''跑 N 次某条路径, 返回 (成功次数, 时延列表)'''
    print(f"\n=== {name} ({runs} 次) ===")
    latencies: list[float] = []
    ok = 0
    for i in range(runs):
        prompt = "What is 137 + 455?"
        expected = {"result": 592}
        success, detail, elapsed = fn(model, prompt, expected)
        latencies.append(elapsed)
        mark = "✓" if success else "✗"
        print(f"  [{i+1}/{runs}] {mark} {detail}")
        if success:
            ok += 1
    rate = ok / runs * 100
    print(f"  --> 成功率: {ok}/{runs} ({rate:.0f}%)")
    return ok, latencies


def check_reproducibility(model: str) -> bool:
    '''复现性验证: 相同输入 5 次, 输出应完全一致 (temperature=0)'''
    print(f"\n=== 复现性验证 (5 次相同输入) ===")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 137+455? Return the sum."}],
        "tools": ADD_TOOLS,
        "tool_choice": ADD_TOOL_CHOICE,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    outputs = []
    for i in range(5):
        resp, err = make_request(payload)
        if err:
            print(f"  [{i+1}] 请求失败: {err}")
            return False
        msg = resp["choices"][0]["message"]
        args = msg.get("tool_calls", [{}])[0].get("function", {}).get("arguments", "") if msg.get("tool_calls") else "<no tool_calls>"
        reasoning_len = len(msg.get("reasoning", "") or "")
        outputs.append((args, reasoning_len))
        print(f"  [{i+1}] args={args}, reasoning_len={reasoning_len}")

    unique_args = set(o[0] for o in outputs)
    unique_reasoning_lens = set(o[1] for o in outputs)
    consistent = len(unique_args) == 1
    print(f"  --> args 唯一值: {unique_args}")
    print(f"  --> reasoning_len 唯一值: {unique_reasoning_lens}")
    print(f"  --> 一致: {consistent}")
    return consistent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL_NAME", "qwen3.6:27b"))
    parser.add_argument("--runs", type=int, default=3, help="每条路径重复次数")
    parser.add_argument("--skip-reproducibility", action="store_true", help="跳过复现性验证 (省时)")
    args = parser.parse_args()

    print(f"模型: {args.model}")
    print(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', '(default)')}")
    print(f"每条路径重复: {args.runs} 次")

    ok_tools, lat_tools = run_path("路径1: tools (tool_calls 路径)", check_tools_path, args.model, args.runs)
    ok_so, lat_so = run_path("路径2: structured output (response_format=json_schema)", check_structured_output_path, args.model, args.runs)
    ok_chat, lat_chat = run_path("路径3: chat fallback (纯 system prompt 指定 JSON)", check_chat_fallback_path, args.model, args.runs)

    reproducible = True
    if not args.skip_reproducibility:
        reproducible = check_reproducibility(args.model)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  路径1 tools             : {ok_tools}/{args.runs}  (平均 {sum(lat_tools)/len(lat_tools):.1f}s)")
    print(f"  路径2 structured output : {ok_so}/{args.runs}  (平均 {sum(lat_so)/len(lat_so):.1f}s)")
    print(f"  路径3 chat fallback     : {ok_chat}/{args.runs}  (平均 {sum(lat_chat)/len(lat_chat):.1f}s)")
    print(f"  复现性 (5次一致)        : {'✓' if reproducible else '✗'}")

    # 判定
    threshold_runs = args.runs * 0.8  # 80% 基线
    all_pass = (ok_tools >= threshold_runs
                and ok_so >= threshold_runs
                and ok_chat >= threshold_runs
                and reproducible)
    print(f"\n判定基线: 每条路径 >= {threshold_runs:.0f}/{args.runs} (80%) 且复现性通过")
    if all_pass:
        print(f"✅ {args.model} 适合作为测试基准 (三条路径均达标 + 复现性通过)")
        return 0
    else:
        failing = []
        if ok_tools < threshold_runs: failing.append(f"tools({ok_tools})")
        if ok_so < threshold_runs: failing.append(f"structured_output({ok_so})")
        if ok_chat < threshold_runs: failing.append(f"chat_fallback({ok_chat})")
        if not reproducible: failing.append("复现性")
        print(f"⚠️  {args.model} 在以下维度未达标: {', '.join(failing)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
