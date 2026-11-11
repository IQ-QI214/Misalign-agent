# QZ 上的 CUA / DinD 准备

日期：2026-11-11。用户明确要求环境搭建全部放在 QZ；Mac 仅用于源码、文档和版本管理。此文件替代此前本机轻量环境准备的执行位置。

## 当前实际状态

- CUA-Lite 作为 Git submodule 固定在 `2af69268c0573f4c12aac391cc03125e6d5c7974`；本机只保留源码，未修改上游文件。
- 本轮曾在 Mac 的项目 `.runtime/` 下新建轻量 Python 环境，用户纠正部署位置后已删除 `cua-bootstrap`、`cua-cpu`、`uv-cache`、`uv-python` 四个目录。没有拉取模型、桌面镜像，没有启动 Docker、GPU 或远端工作。
- 删除前完成了上游核心样本和序列化模块的导入检查；完整模型 adapter 导入因稀疏源码缺少 `lite.gym.utils` 失败。此历史检查不是完整依赖验收，更不是 CUA 环境或训练结果。
- 新增 `scripts/qz_cua_preflight.py`：标准库脚本，只读取系统、挂载、loop 占用及本地 Docker 状态，不安装软件，不启动服务，不挂载或格式化。Mac 平台直接返回未执行状态。
- QZ 实例是否为 Kata、是否支持 Docker、桌面容器是否可运行均未验证。未运行任何真实 GUI 轨迹。

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
| 项目路径 | `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/codex-workspace/Misalign-agent` | 延续已授权 QZ 写入边界 |

若平台不提供 0-GPU 的 Kata 规格，先记录可用的 CPU 开发机配置；不要为环境排障直接占用两张 H100。GPU 透传和 CUDA 支持需要单独确认，不因为 CPU DinD 成功就视为通过。

## 用户需要先做的操作

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
df -h / /inspire/hdd/global_user/wenming-253108090054/zhujiaqi
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
