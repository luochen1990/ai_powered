'''
职责: 私有沙盒后端 (Sandbox) 的离线单元测试 —— 需本机 bubblewrap, 缺失自动整文件跳过.

覆盖 issue #7 安全验收清单 (GenericAgent 方案):
  - 独立 fs: 沙盒内写文件 host 侧读回 / 会话内多次执行共享 / 预置文件可见
  - ls-commands: 命令清单可列出/可 rg 搜索 / 不占用 workspace
  - 隔离面: 宿主 env (含密钥) 不可达 / 宿主 fs 不可见 / 写系统路径必败 / 路径 jail (词法+symlink)
  - 网络: share_net 开 (curl 通) 关 (socket 必败)
  - 配额: 宿主超时 killpg / rlimit CPU·AS·FSIZE
'''
import os
import shutil
import subprocess
import time

import pytest

from ai_powered.sandbox import Sandbox, _COMMANDS_MD_TEMPLATE

requires_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason = "bubblewrap not installed")


@pytest.fixture
def sb():
    s = Sandbox(share_net = False)
    yield s
    s.cleanup()


@pytest.fixture
def net_sb():
    s = Sandbox(share_net = True)
    yield s
    s.cleanup()


def test_workspace_path_jail(sb: Sandbox):
    ''' host 侧文件桥的路径 jail: 词法逃逸与 symlink 越狱均拒 (ValueError 文本回喂的素材) '''
    sb.write_file("data/out.txt", "ok")
    assert sb.read_file("data/out.txt") == b"ok"
    assert sb.list_files() == ["data/out.txt"]
    with pytest.raises(ValueError):
        sb.write_file("../escape.txt", "x")
    with pytest.raises(ValueError):
        sb.read_file("/etc/passwd")
    os.symlink("/etc/passwd", os.path.join(sb.workspace_dir, "leak"))  # 模拟沙盒代码植入的 symlink
    with pytest.raises(ValueError):
        sb.read_file("leak")
    with pytest.raises(ValueError):
        sb.write_file("leak", "pwned")


def test_preset_workspace_files():
    ''' 用法2: workspace_files ClassVar 预置输入, run 期写入沙盒 fs '''
    s = Sandbox(share_net = False, workspace_files = {"input/a.csv": "x,y\n1,2\n", "bin.dat": b"\x00\x01"})
    try:
        assert s.read_file("input/a.csv") == b"x,y\n1,2\n"
        assert s.read_file("bin.dat") == b"\x00\x01"
    finally:
        s.cleanup()


@requires_bwrap
class TestSandboxBwrap:

    def test_basic_execution(self, sb: Sandbox):
        rst = sb.run_bash("echo hello && expr 6 \\* 7", timeout = 30).wait()
        assert rst.exit_code == 0, rst.stderr
        assert "hello" in rst.stdout and "42" in rst.stdout

    def test_file_roundtrip_and_persistence(self, sb: Sandbox):
        ''' 核心需求: 独立 fs 读写 + 会话内跨执行持久 '''
        rst1 = sb.run_bash("echo 'hello sandbox' > data.txt", timeout = 30).wait()
        assert rst1.exit_code == 0, rst1.stderr
        assert sb.read_file("data.txt") == b"hello sandbox\n"  # host 侧读回
        rst2 = sb.run_bash("cat data.txt | tr a-z A-Z", timeout = 30).wait()
        assert rst2.exit_code == 0, rst2.stderr
        assert "HELLO SANDBOX" in rst2.stdout  # 第 2 次执行仍可见 (跨执行共享)

    def test_home_is_workspace(self, sb: Sandbox):
        ''' HOME=/workspace: 工具写入 home 的产物落在 workspace, 随 run 清理 '''
        rst = sb.run_bash("echo x > ~/home.txt && pwd", timeout = 30).wait()
        assert rst.exit_code == 0, rst.stderr
        assert "/workspace" in rst.stdout
        assert sb.read_file("home.txt") == b"x\n"

    def test_ls_commands_available_and_searchable(self, sb: Sandbox):
        ''' ls-commands: 列出清单 / 可 rg 搜索; commands.md 不占用 workspace (用户决策: 不注入 workspace) '''
        rst = sb.run_bash("ls-commands | head -5", timeout = 30).wait()
        assert rst.exit_code == 0, rst.stderr
        assert "Available commands" in rst.stdout
        rst2 = sb.run_bash("ls-commands | rg -c 'rg'", timeout = 30).wait()
        assert rst2.exit_code == 0, rst2.stderr
        assert "commands.md" not in sb.list_files()  # 清单在 /commands (只读挂载), 不在 workspace

    def test_ls_commands_content_matches_ssot(self, sb: Sandbox):
        ''' 清单内容 = 库内 SSOT 渲染 (认知面审核: 改表即生效) '''
        rst = sb.run_bash("ls-commands", timeout = 30).wait()
        assert rst.stdout == _COMMANDS_MD_TEMPLATE

    def test_env_isolated_from_host(self, sb: Sandbox, monkeypatch):
        ''' --clearenv: 宿主 env 中的敏感变量按构造不可达 (密钥隔离) '''
        monkeypatch.setenv("FAKE_API_KEY", "sk-super-secret")
        rst = sb.run_bash("echo $FAKE_API_KEY; echo PATH=$PATH", timeout = 30).wait()
        assert rst.exit_code == 0, rst.stderr
        assert "sk-super-secret" not in rst.stdout
        assert "PATH=" in rst.stdout

    def test_host_fs_invisible(self, sb: Sandbox):
        ''' 宿主 fs 默认不可见: 无 /home /root, workspace 与 /commands 在 '''
        rst = sb.run_bash("ls /", timeout = 30).wait()
        assert rst.exit_code == 0, rst.stderr
        assert "home" not in rst.stdout.split() and "root" not in rst.stdout.split()
        assert "workspace" in rst.stdout.split() and "commands" in rst.stdout.split()

    def test_write_outside_workspace_denied(self, sb: Sandbox):
        ''' 系统路径 ro: 写 /etc 必败 '''
        rst = sb.run_bash("touch /etc/pwned", timeout = 30).wait()
        assert rst.exit_code != 0

    def test_network_disabled_by_default_flag(self, sb: Sandbox):
        rst = sb.run_bash("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)\"", timeout = 15).wait()
        assert rst.exit_code != 0

    def test_network_enabled_with_share_net(self, net_sb: Sandbox):
        ''' share_net=True: 出网可达 (DNS/CA 由 ro-bind 三件套支撑) '''
        rst = net_sb.run_bash("curl -fsS --max-time 20 -o page.html https://example.com && wc -c page.html", timeout = 40).wait()
        assert rst.exit_code == 0, rst.stderr
        assert net_sb.read_file("page.html") != b""  # 下载产物落 workspace

    def test_timeout_kills_process_group(self, sb: Sandbox):
        start = time.monotonic()
        rst = sb.run_bash("sleep 300", timeout = 2).wait()
        elapsed = time.monotonic() - start
        assert rst.timed_out and rst.exit_code != 0 and elapsed < 15

    def test_busy_loop_killed_by_host_timeout(self, sb: Sandbox):
        ''' CPU 自旋 (非 sleep) 同样被宿主超时击杀 —— killpg 对忙轮进程有效 '''
        rst = sb.run_bash("while true; do :; done", timeout = 3).wait()
        assert rst.timed_out

    def test_rlimit_as_memory_cap(self, sb: Sandbox):
        rst = sb.run_bash("python3 -c \"x = 'a' * (2 * 1024 ** 3)\"", timeout = 60).wait()
        assert rst.exit_code != 0 and "MemoryError" in rst.stderr

    def test_rlimit_fsize_cap(self, sb: Sandbox):
        rst = sb.run_bash("head -c $((200 * 1024 * 1024)) /dev/zero > big.bin", timeout = 60).wait()
        assert rst.exit_code != 0

    def test_bwrap_missing_raises_at_run(self, sb: Sandbox, monkeypatch):
        ''' 后端缺失: run 期清晰报错 (构造期探测在 GenericAgent 层) '''
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match = "bubblewrap"):
            sb.run_bash("echo hi", timeout = 10).wait()

    @pytest.mark.asyncio
    async def test_run_bash_does_not_block_event_loop(self):
        ''' async 路径非阻塞: bash 执行期间事件循环仍被调度 (阻塞实现会让 ticker 计数≈0) '''
        import asyncio

        sb = Sandbox(share_net = False)
        try:
            ticks = 0

            async def ticker() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1

            task = asyncio.create_task(ticker())
            rst = await sb.run_bash("sleep 1.2", timeout = 10)
            task.cancel()
            assert rst.exit_code == 0, rst.stderr
            assert ticks > 20  # 1.2s / 0.02s ≈ 60 次期望; 阻塞路径实测为 0
        finally:
            sb.cleanup()


def test_apply_rlimits_sets_soft_equal_hard():
    ''' rlimit 配额的真覆盖: 子进程内应用后回读, 三项 soft==hard 且取值正确 (CPU/AS/FSIZE) '''
    import sys

    import ai_powered.sandbox as sb_mod

    code = (
        "import resource\n"
        f"from ai_powered.sandbox import _apply_rlimits, _RLIMIT_CPU_SECONDS, _RLIMIT_AS_BYTES, _RLIMIT_FSIZE_BYTES\n"
        "_apply_rlimits()\n"
        "print(resource.getrlimit(resource.RLIMIT_CPU)[0] == _RLIMIT_CPU_SECONDS == resource.getrlimit(resource.RLIMIT_CPU)[1])\n"
        "print(resource.getrlimit(resource.RLIMIT_AS)[0] == _RLIMIT_AS_BYTES == resource.getrlimit(resource.RLIMIT_AS)[1])\n"
        "print(resource.getrlimit(resource.RLIMIT_FSIZE)[0] == _RLIMIT_FSIZE_BYTES == resource.getrlimit(resource.RLIMIT_FSIZE)[1])\n"
    )
    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(sb_mod.__file__))}
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "True", "True"]
