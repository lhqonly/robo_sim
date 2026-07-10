# Progress Log

## 2026-07-10 — Phase 1：单关节与恒力矩实验

### 目标

用最小模型建立 `body → joint → actuator → state` 的可运行链路，观察力矩如何改变关节位置和速度；本阶段不引入反馈控制。

### 实现

- `models/single_joint/single_joint.xml`：固定基座、1 DoF hinge、0.5 m / 1 kg 杆件和 `[-2, 2] N·m` motor。
- `experiments/001_single_joint_pd/run.py`：恒力矩实验，支持设置力矩、时长和打印采样数；默认无窗口快速运行，`--view` 在 WSLg 中实时演示动作。
- `tests/test_single_joint.py`：验证模型结构、正力矩产生正向运动，以及 CLI 可独立运行并输出状态列。
- `docs/concepts.md`：补充模型映射、`qpos/qvel/ctrl` 和开环力矩。

### 验证结果

默认实验施加 `0.5 N·m`、运行 1 s：关节从 `0 rad` 运动到约 `0.143473 rad`，最终速度约 `0.078950 rad/s`。Phase 1 新增测试为 `2 passed`，项目完整测试为 `4 passed`；零力矩保持静止，负力矩产生方向相反的运动。当前环境检测到 `DISPLAY=:0`、Wayland 与 `/mnt/wslg`；已实际启动 `--view` 运行 2 s，MuJoCo Viewer 正常显示并实时推进模型。

### 下一步

Phase 2 将保留同一模型，把恒定力矩替换为根据位置/速度误差实时计算的 PD 力矩，并绘制响应曲线比较参数效果。

## 2026-07-10 — 修复不完整虚拟环境

### 现象

运行 `source .venv/bin/activate` 报告文件不存在。随后执行的 `python` 实际来自 `/opt/rk-toolchain/bin/python`，因缺少 `libpython2.7.so.1.0` 失败；`pytest` 则来自用户级全局安装，而不是项目环境。

### 根因与修复

当前 Ubuntu 缺少 `python3.12-venv` 的 `ensurepip` 组件。`python3 -m venv` 在生成激活脚本之前失败，但留下了 Python 软链接，旧版 setup 脚本因此把半成品误判为完整环境。现在 setup 脚本会显式检查 `bin/activate`，缺失时用 `venv --without-pip` 补齐脚本，再定向引导 pip；修复后仍缺失则立即报错。

### 验证

在新的 Bash shell 中激活成功，`python` 和 `pytest` 均指向 `/home/lhq24/robo_sim/.venv/bin/`；环境检查通过，pytest 结果为 `2 passed`。

## 2026-07-10 — Phase 0：工程初始化

### 目标

在 WSL Ubuntu 的 Linux 文件系统中建立可复现、可追溯的 Python/MuJoCo 学习工程。

### 已完成

- 在 `/home/lhq24/robo_sim` 初始化 Git，默认分支为 `main`。
- 创建 `src` 布局、模型、实验、脚本、测试和文档目录。
- 用 `pyproject.toml` 声明 Python 版本、运行依赖和开发依赖。
- 添加一键虚拟环境安装脚本和无窗口环境检查脚本。
- 记录学习路线、基础概念、WSL 环境步骤和首个架构决策。

### 环境发现

- 当前系统 Python 为 3.12.3，满足项目的 Python 3.10+ 要求。
- Git 已安装。
- GitHub CLI (`gh`) 尚未安装；远端仓库已由用户创建，并已通过标准 Git 命令关联为 `origin`：`https://github.com/lhqonly/robo_sim.git`。

### 验证

虚拟环境已经创建，所有依赖已安装。当前 Ubuntu 缺少 `python3.12-venv` 的 `ensurepip` 组件，且当前用户没有免密 sudo；安装脚本会用 `venv --without-pip` 补齐激活脚本，再使用宿主 pip 的 `--python` 选项，只把 pip 引导进项目 `.venv`，没有向系统 Python 安装项目依赖。

实际验收结果：

- Python 3.12.3
- MuJoCo 3.10.0，最小单关节模型编译并完成一步仿真
- NumPy 2.5.1、Matplotlib 3.11.0、SciPy 1.18.0、Jupyter 1.1.1
- pytest 8.4.2：`2 passed`
- `pip check`：`No broken requirements found`
- `git diff --check`：通过

复验命令：

```bash
source .venv/bin/activate
python scripts/check_env.py
pytest
```

### 下一步

进入 Phase 1/2：创建单关节 MuJoCo XML，并在同一实验中逐步加入力矩输入和 PD 控制。开始前不引入 Isaac Lab 或强化学习。
