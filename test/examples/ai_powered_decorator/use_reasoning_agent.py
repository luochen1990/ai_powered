'''
职责: @ai_powered(agent=ReasoningAgent) 的真实 LLM 端到端示例 (test/unit 的离线测试互补).

ReasoningAgent 为多轮推理引擎: 模型可先输出文本推理再调用 return_result 工具提交结果,
适用于需要 step-by-step 的任务 (如多步算术).
'''
from ai_powered import ReasoningAgent, ai_powered


@ai_powered(agent = ReasoningAgent, retry = 1)
def solve_word_problem(problem: str) -> int:
    ''' Solve the word problem step by step and return the numeric answer. '''
    ...


def test_solve_word_problem():
    answer = solve_word_problem("A bakery sold 23 loaves in the morning and 3 times as many in the afternoon. How many loaves were sold in total?")
    print(f"{answer =}")
    assert answer == 23 * 4
