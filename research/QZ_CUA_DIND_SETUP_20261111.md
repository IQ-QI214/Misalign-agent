# QZ 上的 CUA / DinD 准备

日期：2026-11-11。用户明确要求环境搭建全部放在 QZ；Mac 仅用于源码、文档和版本管理。此文件替代此前本机轻量环境准备的执行位置。

## 本轮新实例与下一步

用户已创建 `cua-test-qi` CPU 开发机，并明确将本轮文件与数据的写入目录改为：

```text
/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
```

本次浏览器观察到该实例页面显示开发机模式、40 CPU、400 GB 内存、0 GPU；用户终端输出确认 root、x86_64、cgroup v2，以及 Docker client/daemon 29.2.0 已安装。`/var/run/docker.sock` 不存在，`losetup -a` 未列出绑定。这是初检状态；后续 daemon 与小容器均已通过，见下方最新验收。

用户检查中的 460G 可用空间属于旧 `global_user` 挂载；**新 project 挂载的剩余空间仍未知**。旧 CPU SSH 中转能够连接，但看不到新项目挂载，不能在旧机器上创建同名目录代替它。浏览器原生接口可以读取新页面，输入操作却未产生可验证的终端变化；后续通过新实例的 HTTPS/WebSocket SSH 连接完成了远端操作。

新增 `scripts/qz_cua_dind.py`，在新实例内执行：

```bash
python3 -B scripts/qz_cua_preflight.py
python3 -B scripts/qz_cua_dind.py --apply --kata-confirmed
```

不带 `--apply` 只打印计划；在 Mac 上即使带 `--apply` 也拒绝执行。`--kata-confirmed` 指本次用户已选择的 QZ 开发机模式，不能在普通共享 CPU 节点上使用。

执行版遵循图片中的 cgroup v2 → ext4 loop → overlay2 路线，并增加以下资源边界：

- 持久数据、配置、daemon 日志和逐次 JSON 回执位于 `zjq/.runtime/cua-dind/<hostname>/`。实例内的 `/run`、loop 设备节点及 cgroup 是启动所需的临时系统状态。
- 初始镜像逻辑容量 30 GiB（按平台手册示例）；新建前要求项目挂载至少 46 GiB 可用。这是初始工程容量与余量，不是 CUA 完整运行预算，也不是底层已预留 30 GiB。已有镜像不扩容、不重新格式化。
- 只对以排他方式新建的普通文件运行 `mkfs.ext4`，从不把 `/dev/loopN` 作为格式化目标。使用空闲 loop 和 `--nooverlap`；挂载之后再核实 backing file 身份。
- 检查已有 daemon、socket、实例 boot ID 和存储所有权。冲突时停止；失败时保留文件、挂载和日志以便排查，不进行全局清理或盲目重启。
- 显式选择 classic `overlay2`，关闭 containerd snapshotter，使用项目 data-root；仅监听实例 Unix socket。与平台全局 `/etc/docker/daemon.json` 分开，不覆盖它。
- daemon 通过后运行一个官方 `hello-world` 小容器，限制为 1 CPU、64 MiB、无容器网络；记录实际 image ID、退出状态与输出。镜像下载失败和 daemon 失败分别记录。

最终只有出现 `DIND_SMOKE_PASSED_GUI_NOT_RUN` 才表示该次 Docker 小容器验收通过；它不代表 GUI 环境、截图、动作、状态重建或训练已打通。脚本失败时贴出最后的 JSON 与 `RECEIPT=...`，保留现场。

## 已有 WeaveBench VM 的位置

已查阅用户指定的《排查 WeaveBench 虚拟机下载失败》会话。历史完成记录为 2026-11-07 03:01（北京时间）：

```text
/inspire/hdd/project/independent-security-research/wenming-253108090054/czk/weavebench/cache/vm/Ubuntu.qcow2
bytes: 28457500672
sha256: 3c1caf41cb75b8482a30d1a251545393aee8175375ab405e0d6870f8a07fa3f8
receipt: ../vm-aria2/status.json
```

这是历史校验结果；当前新实例中的文件存在性和内容尚未重新核验。只读 preflight 会检查存在性/大小，不将旧 SHA 当成当前已验证的 SHA。该资产不属于本轮写入目录，不能直接启动成可写磁盘；如以后需要，先核验，再在 `zjq` 内创建独立 overlay。

第一步采用固定版本 CUA-Lite 的 **Lite.OSWorld（VM-free Docker desktop）**，因此目前不需要重新下载或复制这份约 26.50 GiB 的 VM，也不需要为此先解决嵌套 KVM。源码依据：`references/cua_lite` commit `2af69268c0573f4c12aac391cc03125e6d5c7974` 的 `lite/gym/envs/lite/osworld/README.md`。

## 当前实际状态

- CUA-Lite 作为 Git submodule 固定在 `2af69268c0573f4c12aac391cc03125e6d5c7974`；本机只保留源码，未修改上游文件。
- 本轮曾在 Mac 的项目 `.runtime/` 下新建轻量 Python 环境，用户纠正部署位置后已删除 `cua-bootstrap`、`cua-cpu`、`uv-cache`、`uv-python` 四个目录。没有拉取模型、桌面镜像，没有启动 Docker、GPU 或远端工作。
- 删除前完成了上游核心样本和序列化模块的导入检查；完整模型 adapter 导入因稀疏源码缺少 `lite.gym.utils` 失败。此历史检查不是完整依赖验收，更不是 CUA 环境或训练结果。
- 新增 `scripts/qz_cua_preflight.py`：标准库脚本，只读取系统、挂载、loop 占用及本地 Docker 状态，不安装软件，不启动服务，不挂载或格式化。Mac 平台直接返回未执行状态。
- 已有新实例页面与用户终端的部分基础检查；Docker daemon 和官方 hello-world 容器已通过执行验收；桌面与真实 GUI 轨迹待验证。

## 用户提供图片中的安装路线

图片为《启智开发机用户手册》。已读取原图；原图不上传公开仓库。

1. 创建交互式建模，选择“开发机模式”（手册提示只有开启配置的空间才可见）。
2. 选择 `dind:1.0.0` 镜像；该镜像已包含 Docker 与相关脚本。
3. 分配较充足的内存。Kata 有独立 Guest OS，手册说明有额外内存开销。
4. 在 Notebook 终端中准备 cgroup v2；为 Docker 数据准备 loop-backed ext4；启动 Docker daemon；检查 `docker ps`。
5. 图片下半部分为自制 DinD 镜像的 Dockerfile，本项目使用平台现成镜像时无需执行。

## 第一台实例的建议配置

| 项目 | 建议 | 依据和限制 |
| --- | --- | --- |
| 用途 | CUA 环境、重放、验证与数据准备 | 先验证桌面环境，不分配模型训练卡 |
| 模式 | 开发机模式 / Kata | 按用户提供手册 |
| 镜像 | `dind:1.0.0` | 按用户提供手册，不在普通 notebook 里反复安装 Docker |
| GPU | 0 | 当前环境验收不需要 GPU；之后模型推理和训练使用独立 GPU 实例 |
| CPU / RAM | 优先 8 vCPU / 32 GiB；先 1 个桌面容器 | 工程起步建议，不是平台硬性要求或已测资源预算 |
| 存储 | 先核对共享盘可用空间，再分配 Docker 镜像/容器预算 | 手册 30G 是示例，不能当完整 CUA 与训练镜像所需空间保证 |
| 项目路径 | `/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq/Misalign-agent` | 用户本轮明确指定的新写入目录 |

若平台不提供 0-GPU 的 Kata 规格，先记录可用的 CPU 开发机配置；不要为环境排障直接占用两张 H100。GPU 透传和 CUDA 支持需要单独确认，不因为 CPU DinD 成功就视为通过。

## 基础只读检查（首轮已部分完成）

创建上述实例，打开 Terminal，执行下面的只读检查。代码可直接粘贴；没有安装、挂载、格式化、启动 Docker 或模型的命令。

```bash
id
uname -m
printf '\n--- cgroup ---\n'
stat -fc %T /sys/fs/cgroup
cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null
printf '\n--- commands ---\n'
command -v docker dockerd dockerd-entrypoint.sh losetup mkfs.ext4
printf '\n--- disk ---\n'
df -h / /inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
printf '\n--- loop devices ---\n'
losetup -a
printf '\n--- local docker ---\n'
docker --host unix:///var/run/docker.sock version
```

或者，若项目已同步到本次提交，在项目目录执行更完整的结构化检查：

```bash
python3 scripts/qz_cua_preflight.py
```

用户提供新实例 IDE 链接与检查输出后，继续核实该实例身份和连接方式。旧 `connect_gpu.sh` 明确要求两张 GPU，不能直接用于这个 CPU 实例；后续如需中转，应为新 CPU 实例单独配置并核验，不重放旧 GPU controller。

## 启动 Docker 时如何使用手册

手册中的 cgroup v2、ext4 loop 与 daemon 启动路线可作为基线；实际启动脚本要根据新实例检查结果确定：

- 先检查是否已有正常 daemon；已有服务不重启、不覆盖其数据。
- 选择空闲 loop 设备，不固定占用 `/dev/loop0`；新建本项目 backing image 才初始化文件系统，已有 image 不重新 `mkfs`。
- 数据和日志优先放本项目授权工作区，避免把大型镜像缓存留在不明确的系统盘位置。rootfs、共享存储和 loop 文件内部容量分别记录；稀疏文件标称容量不等于已经预留了底层磁盘。
- 只在确认的 Kata 开发机内做手册要求的 cgroup/loop 初始化，不把这类系统修改带到普通共享 CPU 中转机。
- 只监听实例本地 Unix socket。根据实际 Docker 版本、镜像 daemon 配置和存储驱动决定参数；不混用图片最后的 fuse-overlayfs 简化例与前文 ext4/overlay2 配方。
- `docker ps` 成功仅证明 daemon 可访问。下一步还要创建一个本项目容器，验证读写、重启/重置和网络，然后才部署一个 CUA-Lite 桌面容器、拿到截图并执行参考操作。

参考：[Docker daemon 配置](https://docs.docker.com/engine/daemon/)、[dockerd 参数](https://docs.docker.com/reference/cli/dockerd/)、[losetup 参数](https://man7.org/linux/man-pages/man8/losetup.8.html)。官方文档确认 data-root 可配置、重复配置可能导致启动失败，以及同一 backing file 的重复 loop 绑定可能导致损坏；它们不替代平台的 Kata 使用要求。

## 后续实施顺序

实例只读核验 → 项目专用 DinD 启动 → 一个真实桌面容器与截图/动作验收 → 固定训练来源任务与安全验证器 → 状态重建和分支记录 → GPU 推理/转化 → 训练与独立评测。

环境部署位置调整不改变研究问题。无效攻击经历怎样转为有效训练监督，仍需最终通过训练后的 CUA 行为结果证明；本轮只有源码固定和 QZ 操作准备。

用户首轮真实执行回执显示项目挂载剩余 81,906,368,512 bytes（约 76.28 GiB），旧默认 64 GiB + 16 GiB 余量触发容量检查，尚未创建数据镜像或启动 daemon。已将默认值改为 30 GiB；现有脚本也可直接传 `--size-gib 30` 接续，无需重新 clone。VM 文件存在且大小符合历史记录，当前 SHA 未检查。

## 最新验收（2026-11-11）

已通过用户提供的 30003 HTTPS 代理接通 SSH，并核对用户提供的 ED25519 指纹。旧中转到新实例内网的 TCP 连接超时。现有 rtunnel 客户端使用额外数据包格式，与本次标准二进制 WebSocket 桥接不兼容，因此采用 `scripts/qz_cua_ws_client.py` 作为 OpenSSH ProxyCommand。客户端仅使用 Python 标准库，TLS 验证和 SSH 严格主机密钥检查均开启。访问 URL、known_hosts 和本机调用入口放在忽略的 `.runtime/qz/` 中。

新连接检查时根 cgroup 已无驻留进程，重试原 `prepare_cgroups()` 成功。原 EBUSY 未捕获具体调用位置，故保留历史失败，不将重试成功当作已证明某个唯一根因。

Docker 29.2.0、cgroup v2、overlay2 与项目专用 ext4 已启用。首次 hello-world 镜像配置下载被重置，失败回执保留；复用 daemon 和已下载 layer 重试成功，容器 ExitCode=0。30 GiB backing file 实际占用 520,093,696 bytes，项目挂载剩余 81,352,720,384 bytes。完整可公开验收摘要见 `validation/cua_qz_dind_acceptance.json`。

CUA-Lite 源码已在 QZ 检出固定 commit，正在进行 CPU 环境部署。小容器验收、源码就绪和真实 GUI 任务分别记录，后续仍需桌面截图、动作与任务重置验证。

## 存储管理

按用户最新说明采用常规总量检查，保留运行必需内容，及时清理明显冗余缓存。代码、配置、脱敏结果及重建记录随批次推送 GitHub；连接凭据保存在私有目录。后续 CPU 依赖安装使用 `--no-cache`，源码使用 cone sparse checkout 保留 `lite`、`scripts` 与根目录元数据。QZ 的 Git 版本需要先 `git sparse-checkout init --cone`，再 `git sparse-checkout set lite scripts`。

初次存储审计按 `du -sx -B1 zjq` 统计实际分配量，避免重复计算 ext4 mount 与 backing file。清理前 1,082,491,904 bytes，清理后 1,015,703,040 bytes，已清理 66,788,864 bytes。清单见 `validation/cua_qz_storage_v1.json`。此后若安装依赖或镜像，需要更新实际占用。

Python 3.12.16 与 uv 0.13.7 已安装在 `zjq/.runtime/cua-host`。PyPI 访问超时的日志已保留，切换到可访问的阿里云 PyPI 镜像。`scripts/qz_cua_gym_install.sh` 提供仅安装 `[gym]` 的 CPU 主机依赖入口。完整桌面仍需镜像、环境专用依赖及真实动作验证。GHCR 元数据预检显示 Lite.OSWorld 镜像的压缩 layers 合计 2,579,478,557 bytes，解压后的实际占用需要拉取后测量。
