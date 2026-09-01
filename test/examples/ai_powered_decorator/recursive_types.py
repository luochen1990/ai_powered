'''
职责: 递归类型的 e2e 示例 —— capability-gated schema refs 透传的手动验证用例 (issue 18).

递归 Struct (TreeNode) 的生成 + 结构往返校验 (期望树与生成树的逐层 match 判定).
仅在支持 schema_refs 能力的 provider 上收集到测试 (如 `just test-provider openai`);
无该能力的 provider (如默认 ollama) 下, @ai_powered 装饰会在递归返回类型上抛
RecursiveSchemaError —— 而装饰发生在模块 import (= pytest collection) 期, 会破坏收集,
故守卫是模块级 if 而非 skipif (后者求值太晚), 对齐 use_generic_agent.py 的守卫惯例.
经 OPENAI_MODEL_FEATURES 注入 schema_refs 而后端实际不支持时, 失败模式为运行期 provider 报错
(全局纠错出口的固有语义: 声明错位不在本地校验, 由 provider 实际响应兜底暴露).
TreeNode / 判定函数为纯数据结构无门控, 保持在 if 块外.
'''
import msgspec

from ai_powered import ai_powered
from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.resolver import resolve_model


class TreeNode(msgspec.Struct):
    name: str
    children: list['TreeNode'] = []


def tree_match(expect: TreeNode, got: TreeNode) -> bool:
    ''' judge if two trees are consistent: name equality + same children count + children recursion '''
    if expect.name != got.name or len(expect.children) != len(got.children):
        return False
    return all(tree_match(e, g) for e, g in zip(expect.children, got.children))


def expected_tree(depth: int) -> TreeNode:
    ''' 期望树的程序化构造 (判定基准不依赖 LLM 输出格式) '''

    def build(path: str, level: int) -> TreeNode:
        if level == depth:
            return TreeNode(path, [])
        return TreeNode(path, [build(f"{path}/x", level + 1), build(f"{path}/y", level + 1)])

    return build("root", 0)


if ModelFeature.schema_refs in resolve_model(None).model_config.supported_features:

    @ai_powered
    def gen_sample_file_tree(depth: int) -> TreeNode:
        '''
        generate a sample file tree with the given depth,
        each non-leaf node has exactly 2 children named "x" and "y",
        and every node name is the slash-joined path from the root (root name is "root")
        '''
        ...

    def test_gen_sample_file_tree():
        tree = gen_sample_file_tree(2)
        print(f"{tree =}")

        assert tree_match(expected_tree(2), tree)
