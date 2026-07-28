# LinkJoin Robo SIM

LinkJoin Robo SIM 是一个面向初学者的学习型工程：用仿真逐步建立“人体 + 下肢外骨骼 + 助力控制 + 数据分析”的最小闭环。

项目从轻量的 MuJoCo 和单关节模型开始，先学习关节、自由度、力矩与反馈控制，再逐步扩展到单腿、膝关节助力、简化人体和数据记录。当前不引入 Isaac Sim、Isaac Lab 或强化学习。

## 当前状态

Phase 1（单关节恒力矩）、Phase 2（PD 闭环控制）、Phase 2.5（重力补偿）和 Phase 2.6（响应时间）已完成：可以在原生 MuJoCo Viewer + 中文面板中观察反馈控制，实时开关重力补偿，并测量首次到达、上升时间、稳定时间和最大超调。下一阶段是把当前杆明确映射为膝关节/小腿摆杆。

权威路线图见 [`docs/work_plan.md`](docs/work_plan.md)，完成记录见 [`docs/progress_log.md`](docs/progress_log.md)。

## 环境要求

- WSL2 Ubuntu 或原生 Ubuntu
- Python 3.10 及以上（当前环境实测为 Python 3.12）
- Git

仓库应放在 WSL Linux 文件系统（例如 `~/robo_sim`），不要放在 `/mnt/c`。这样通常有更好的文件 I/O 性能，也能避免 Linux 权限与 Windows 文件语义之间的问题。

## 快速开始

```bash
cd ~/robo_sim
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py
python -m pytest
```

环境检查会验证 Python 版本、核心依赖、项目包导入，以及 MuJoCo 是否能编译一个最小模型；它不会打开图形窗口。

运行无窗口 PD 闭环实验并查看数值：

```bash
python experiments/001_single_joint_pd/run.py \
  --mode pd --target-deg 30 --kp 30 --kd 3 \
  --duration 3 --samples 7
```

在 WSLg 中打开 Viewer 和网页动态响应曲线：

```bash
python experiments/001_single_joint_pd/run.py \
  --mode pd --view --target-deg 45 --kp 5 --kd 3 \
  --gravity-compensation
```

`--view` 会同时打开官方 3D Viewer 和中文学习控制面板。PD 模式下，参数、稳定判据、角度曲线和响应指标集中在同一个调参区域；蓝色按钮会应用参数并自动从 `0°` 开始一轮可比较测试。响应时间默认以 `±1°` 且连续保持 `0.5 s` 判定稳定，显示首次到达、上升时间、稳定时间和超调。动态曲线还会拆分 PD、重力补偿和实际总力矩。`--mode torque` 仍保留精确扭矩输入、滑块和预设按钮。

## 目录导览

```text
docs/          学习路线、概念、进展和架构决策
src/robo_sim/ 可复用的控制器、模型辅助代码和工具
models/        MuJoCo XML 模型
experiments/   可独立运行、可复现实验
scripts/       环境安装和检查脚本
tests/         自动化测试
```

## 学习方式

每一阶段都回答四个问题：目标是什么、涉及什么概念、代码如何实现、结果如何验证。每次路线变化都应记录在 `docs/progress_log.md` 或 `docs/decisions/` 中。
