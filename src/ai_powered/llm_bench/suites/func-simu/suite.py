# func-simu suite 元数据
#
# 职责边界: 套件名 / 种子轮换粒度 / 套件描述; 用例在本目录其余 *.py 文件中

from ai_powered.llm_bench import SuiteMeta

SUITE_META = SuiteMeta(
    name="func-simu",
    rotation="day",
    description="函数模拟套件: 让模型按类型化函数签名生成输出 (初始任务显式转化自 ai_powered@a1a39e0)",
)
