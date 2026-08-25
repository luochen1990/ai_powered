'''
职责: GenericAgent 的私有沙盒后端 —— bwrap mount-namespace 隔离 + scratch dir workspace (不导出公共 API).

安全分层 (参考 ~/ws/nixos/apps/sandbox-exec 同款骨架):
  1. namespace 隔离: --unshare-all [+ --share-net]
  2. capability 剥夺: --cap-drop ALL
  3. 环境清空: --clearenv + 显式重建 PATH/HOME/TMPDIR (env 密钥按构造不可达)
  4. 文件系统限制: 宿主 fs 默认不可见, 仅显式 bind 的路径存在; scratch dir 独占 rw bind 为 /workspace;
     网络模式补 CA/DNS 只读挂载 (realpath 解析到实际路径)
  5. 资源配额: child 侧 preexec setrlimit (CPU/AS/FSIZE, 跨 fork/exec 继承 → 整棵沙盒树受限;
     刻意不含 NPROC —— 它按宿主真实 UID 计数, preexec 设置会让 bwrap 自身 clone EAGAIN)
  6. 宿主兜底: 超时 killpg + --die-with-parent; fork 炸弹类失控由宿主超时击杀 + pid-ns init
     死亡传播兜底 (pid namespace 不限量进程数, 只保证 init 死则全树死)

命令清单是认知面审核 (非约束面): 清单管"模型被告知什么", 实际可用命令由沙盒环境决定 ——
清单外命令即使被模型猜中调用也出不了 bwrap 边界 (与 sandbox-exec 的 PATH 哲学一致).

已知边界 (诚实标注):
  - mount-namespace 级隔离, 防误操作与 prompt 注入诱导的破坏性代码, 非对抗 dedicated attacker
    的硬边界 (与 Flatpak 同级); share_net 下数据外渗/供应链风险由文档标注
  - 非 NixOS 分支按需 bind 系统路径 (/usr /lib* /bin /sbin + passwd/group 供 uid 查询),
    不整绑 /etc (其下的 profile.d 等可能含凭据); 但该分支无法在 NixOS CI 上测试, 属尽力而为
'''
import asyncio
import os
import resource
import shutil
import signal
import subprocess
import tempfile
from typing import Mapping

from easy_sync import sync_compatible
import msgspec


class ExecutionResult(msgspec.Struct):
    ''' 一次沙盒执行的回执: 执行错误是回喂模型自修复的素材, 不烧 retry 预算 '''
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


#: 认知面审核清单 (SSOT): 渲染为 commands.md 与 bash 工具 docstring 简表; rg 替代 grep (用户决策)
COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("files & streams", ("ls", "cat", "head", "tail", "wc", "cp", "mv", "rm", "mkdir", "touch", "ln", "stat", "file", "basename", "dirname", "find", "diff", "tee")),
    ("text processing", ("rg", "sort", "uniq", "cut", "tr", "xargs", "awk", "sed")),
    ("data & compute", ("jq", "python3")),
    ("archives", ("tar", "gzip", "unzip")),
    ("network", ("curl", "wget")),
    ("misc", ("bash", "echo", "date", "which", "ls-commands")),
)

_COMMANDS_MD_TEMPLATE = (
    "# Available commands in this sandbox\n\n"
    "The sandbox isolates the filesystem and network, not the command set. "
    "Commands below are known-good; anything else present in the environment may also work.\n\n"
    + "\n".join(f"## {group}\n" + ", ".join(f"`{c}`" for c in cmds) + "\n" for group, cmds in COMMAND_GROUPS)
    + "## Tips\n"
    "- `ls-commands` prints this file; search it with e.g. `ls-commands | rg json`\n"
    "- use `rg` (not grep) for text search; `python3 -c '...'` for computation\n"
)

#: ls-commands shim 脚本 (shebang 用构造时解析的 bash 绝对路径 —— 空根沙盒内无 /bin/sh 兜底)
_LS_COMMANDS_TEMPLATE = "#!{bash}\nexec cat /commands/commands.md\n"

#: rlimit 配额 (child 侧 preexec 设置; 跨 fork/exec 继承, 整棵沙盒树受限).
#: 刻意不含 RLIMIT_NPROC —— nproc 按宿主真实 UID 计数 (非进程树), preexec 期设置会让 bwrap 自身的
#: clone 全部 EAGAIN; fork 炸弹由 pid namespace + 宿主超时兜底 (与 sandbox-exec 参考实现一致: 仅 rss/cpu/fsize)
_RLIMIT_CPU_SECONDS = 600      # 远高于 bash timeout: rlimit 只兜底失控循环, 常规超时由宿主 killpg 管
_RLIMIT_AS_BYTES = 1 << 30     # 1 GiB
_RLIMIT_FSIZE_BYTES = 128 << 20


def _apply_rlimits() -> None:
    ''' preexec_fn: 在 bwrap exec 前设 rlimit (软=硬, 升硬需 CAP_SYS_RESOURCE, 沙盒内不可逆) '''
    for which, limit in (
        (resource.RLIMIT_CPU, _RLIMIT_CPU_SECONDS),
        (resource.RLIMIT_AS, _RLIMIT_AS_BYTES),
        (resource.RLIMIT_FSIZE, _RLIMIT_FSIZE_BYTES),
    ):
        resource.setrlimit(which, (limit, limit))


def _resolve_sandbox_path() -> str:
    ''' 沙盒 PATH: NixOS 下将继承 PATH 条目的符号链接解析到 /nix/store 实际路径 (store 已 ro-bind, 全覆盖);
    常规发行版用基础系统路径 (已在 ro-bind 集内) '''
    if not os.path.isdir("/nix"):
        return "/usr/bin:/bin"
    parts = []
    for entry in os.environ.get("PATH", "").split(":"):
        if not entry:
            continue
        if entry.startswith("/nix/store/"):
            parts.append(entry)
        else:
            real = os.path.realpath(entry)
            parts.append(real if real.startswith("/nix/store/") else entry)
    return ":".join(parts)


def _resolve_ca_bundle() -> str | None:
    ''' CA 证书解析为实际路径 (NixOS 符号链接 → store); 网络模式下经 env 指给 OpenSSL/curl '''
    for name in ("ca-certificates.crt", "ca-bundle.crt"):
        path = f"/etc/ssl/certs/{name}"
        if os.path.exists(path):
            return os.path.realpath(path)
    return None


def _run_bash_via(self: "Sandbox", script: str, timeout: float) -> ExecutionResult:
    ''' sync 路径 trampoline: 经实例动态派发 —— sync_fn 若直接捕获基类方法, 子类覆盖 _run_bash_blocking
    会被 sync 路径静默绕过 (闭包 vs self 动态查找的分叉), 与 async 路径行为不一致 '''

    return self._run_bash_blocking(script, timeout)


class Sandbox:
    ''' 单次 agent run 的沙盒会话: scratch dir (workspace + meta) + bwrap 执行.

    结构: <scratch>/workspace → 沙盒内 /workspace (rw, cwd, HOME);
          <scratch>/meta      → 沙盒内 /commands (ro, commands.md + ls-commands shim, PATH 前置其 bin).
    生命周期 = 单次 run (run 内多次执行共享 fs, finally cleanup).
    '''

    root: str

    def __init__(self, *, share_net: bool, workspace_files: Mapping[str, str | bytes] = {}) -> None:
        self.share_net = share_net
        self.root = tempfile.mkdtemp(prefix = "ai_powered_sandbox_")
        self.workspace_dir = os.path.join(self.root, "workspace")
        self.meta_dir = os.path.join(self.root, "meta")
        os.makedirs(self.workspace_dir)
        os.makedirs(os.path.join(self.meta_dir, "bin"))
        with open(os.path.join(self.meta_dir, "commands.md"), "w", encoding = "utf-8") as f:
            f.write(_COMMANDS_MD_TEMPLATE)
        bash = os.path.realpath(shutil.which("bash") or "/bin/bash")  # 空根沙盒内无 /bin/sh, shebang 须绝对路径
        shim = os.path.join(self.meta_dir, "bin", "ls-commands")
        with open(shim, "w", encoding = "utf-8") as f:
            f.write(_LS_COMMANDS_TEMPLATE.format(bash = bash))
        os.chmod(shim, 0o755)
        try:  # 预置输入 (用法2: 子类 ClassVar); 越界路径等配置错误在这里爆 —— 清理 scratch dir 再上抛, 不泄漏
            for name, content in workspace_files.items():
                self.write_file(name, content)
        except Exception:
            self.cleanup()
            raise

    # ---------------- host 侧文件桥 (路径 jail: 词法 + realpath 双检) ----------------

    def _resolve(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError(f"workspace file name must be a non-empty string, got {name!r}")
        path = os.path.normpath(os.path.join(self.workspace_dir, name))
        if path != self.workspace_dir and not path.startswith(self.workspace_dir + os.sep):
            raise ValueError(f"workspace path escapes the sandbox root: {name!r}")
        real = os.path.realpath(path)  # 解析已存在前缀上的符号链接 (含中间目录), 堵沙盒代码植入的 symlink 越狱
        if real != self.workspace_dir and not real.startswith(self.workspace_dir + os.sep):
            raise ValueError(f"workspace path resolves outside the sandbox root (symlink escape?): {name!r}")
        return path

    def write_file(self, name: str, content: str | bytes) -> None:
        path = self._resolve(name)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok = True)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        with open(path, "wb") as f:
            f.write(payload)

    def read_file(self, name: str) -> bytes:
        with open(self._resolve(name), "rb") as f:
            return f.read()

    def list_files(self) -> list[str]:
        files: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(self.workspace_dir):
            files.extend(os.path.relpath(os.path.join(dirpath, fn), self.workspace_dir) for fn in filenames)
        return sorted(files)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors = True)

    # ---------------- bwrap 执行 ----------------

    def _build_command(self, script: str) -> list[str]:
        ''' bwrap argv (bwrap 默认空 root: 宿主 fs 按构造不可见, 仅显式 bind 的路径存在) '''
        cmd: list[str] = ["bwrap", "--unshare-all", "--die-with-parent", "--cap-drop", "ALL", "--clearenv", "--new-session"]
        if self.share_net:
            cmd += ["--share-net"]
        if os.path.isdir("/nix"):  # NixOS: 只绑 store (与 sandbox-exec 一致), 命令经 PATH 解析到 store 路径
            cmd += ["--ro-bind", "/nix/store", "/nix/store"]
        else:  # 常规发行版: 按需 bind (刻意不整绑 /etc —— profile.d 等可能含凭据; passwd/group 供 uid 查询)
            for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/passwd", "/etc/group"):
                cmd += ["--ro-bind-try", p, p]
        if self.share_net:
            # CA/DNS: 用 realpath 解析后的真实路径作源 (NixOS 的 /etc/resolv.conf → /run/systemd/resolve/...,
            # 直接 bind 符号链接会因沙盒内目标悬空而失败); --share-net 共享宿主 netns, stub-resolv 的 127.0.0.53 可达
            for src, dst in (("/etc/ssl/certs", "/etc/ssl/certs"), ("/etc/resolv.conf", "/etc/resolv.conf"), ("/etc/hosts", "/etc/hosts")):
                real = os.path.realpath(src)
                if os.path.exists(real):
                    cmd += ["--ro-bind", real, dst]
        cmd += [
            "--bind", self.workspace_dir, "/workspace",
            "--ro-bind", self.meta_dir, "/commands",
            "--chdir", "/workspace", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--setenv", "PATH", f"/commands/bin:{_resolve_sandbox_path()}",
            "--setenv", "HOME", "/workspace",
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "LANG", "C.UTF-8",
            "--setenv", "LC_ALL", "C.UTF-8",
        ]
        if self.share_net:
            ca = _resolve_ca_bundle()
            if ca:
                cmd += ["--setenv", "SSL_CERT_FILE", ca, "--setenv", "CURL_CA_BUNDLE", ca]
        return cmd + ["bash", "-c", script]

    def _run_bash_blocking(self, script: str, timeout: float) -> ExecutionResult:
        ''' 阻塞实现: Popen + communicate + 超时 killpg (sync 路径与 to_thread 共用) '''
        if shutil.which("bwrap") is None:
            raise RuntimeError("bwrap is not installed; bubblewrap is required to run the sandbox")
        proc = subprocess.Popen(  # noqa: S603 无 shell 二次展开 (bwrap argv 直传), 脚本经 bash -c
            self._build_command(script), stdin = subprocess.DEVNULL, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
            start_new_session = True, preexec_fn = _apply_rlimits,
        )
        try:
            out_b, err_b = proc.communicate(timeout = timeout)
            exit_code, timed_out = (proc.returncode if proc.returncode is not None else -1), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:  # 竞态: 超时与击杀之间进程已自行退出
                pass
            out_b, err_b = proc.communicate()
            exit_code, timed_out = -signal.SIGKILL, True
        return ExecutionResult(out_b.decode("utf-8", errors = "replace"), err_b.decode("utf-8", errors = "replace"), exit_code, timed_out)

    @sync_compatible(sync_fn = _run_bash_via)  # 方法形态: sync_fn 经 *args 收到 self, 转发至动态派发
    async def run_bash(self, script: str, timeout: float) -> ExecutionResult:
        ''' 执行 bash 脚本 (cwd=/workspace); timeout 由调用方 clamp 后传入, 到期 killpg 整树击杀.

        async 路径经 to_thread 卸载阻塞实现 —— 事件循环在 bash 执行期间保持可调度
        (sync 路径经 _run_bash_via 直达阻塞实现, 双实现形态见 easy_sync sync_compatible 用法 2)
        '''
        return await asyncio.to_thread(self._run_bash_blocking, script, timeout)
