import pytest
from pydantic.dataclasses import dataclass
from typing import Optional
from ai_powered import ai_powered

@dataclass
class UserInfo:
    name: str
    country: Optional[str]
    age: Optional[str]

@ai_powered
def gen_user_info() -> UserInfo:
    ''' ramdomly generate user info for testing, None is used for unknown fields '''
    ...

@ai_powered
def gen_self_introduction_for(user_info: UserInfo) -> str:
    ''' generate self introduction based on user info '''
    ...

@ai_powered
def extract_user_info(raw_text: str) -> UserInfo:
    ...

def test_gen_user_info():
    user_info = gen_user_info()
    print(f"{user_info =}")

def test_gen_self_introduction():
    user_info = gen_user_info()
    self_introduction = gen_self_introduction_for(user_info)
    print(f"{self_introduction =}")

# LLM 往返一致性测试: 输入本身由模型生成 (每轮不同), None 字段的序列化编码存在固有
# 歧义, 是免费档基准模型的边缘能力 (GitHub CI 基线 nemotron-120b:free temp=0 下
# 失败率个位数, 未控温时 ~10-30%; forgejo CI 基线 2026-09-01 起为 sg default-plan
# flash (glm-5.3-flash), 两次全量 (-n2/-n0) 均首发通过; 历史基线 (免费池
# ultra/flash 档) 的边缘性与失败数据见 .forgejo/workflows/ci.yml 选型史与
# EXTRA_BODY 注)。
# reruns=3 与 forgejo CI 全局持平 (比 GitHub CI 全局多一次); delay 20s 对齐 forgejo
# 时间隔离策略。marker 优先级高于 CLI, 缺省会回落 0 而非继承 CLI, 故 delay 必须显式;
# marker 跨 CI 生效, GitHub 侧 delay 同抬至 20s (最坏 +60s, 属可接受代价)。容忍采样
# 噪声与上游瞬态故障, 连续失败仍判红 (不掩盖真回归)。
@pytest.mark.flaky(reruns=3, reruns_delay=20)
def test_extract_identity():
    user_info = gen_user_info()
    self_introduction = gen_self_introduction_for(user_info)
    user_info_extracted = extract_user_info(self_introduction)
    assert user_info_extracted == user_info
