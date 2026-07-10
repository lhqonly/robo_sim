# LinkJoin Robo SIM Work Plan

本文档是项目路线的唯一权威来源。状态变化记录在 [`progress_log.md`](progress_log.md)，重要路线调整另写 ADR 到 `docs/decisions/`。

## 项目目标

围绕消费级下肢外骨骼，建立以下最小闭环：

```text
简化人体/腿部模型 → 关节控制 → 外骨骼助力 → 仿真数据记录与分析
```

工程遵循 Ubuntu-first、WSL2-compatible、MuJoCo-first、控制优先于强化学习的原则。模型复杂度依次为：

```text
单关节 → 摆杆 → 单腿 → 双腿 → 简化人体 → 人体 + 外骨骼
```

## 阶段总览

| 阶段 | 状态 | 核心目标 | 主要产物 |
| --- | --- | --- | --- |
| Phase 0 | 已完成 | WSL/Ubuntu 工程初始化 | 工程骨架、环境脚本、基础文档 |
| Phase 1 | 已完成 | 理解机器人/外骨骼最小单元 | 单关节 XML、读取状态并施加力矩 |
| Phase 2 | 待开始 | 理解反馈与 PD 控制 | 通用 PD 控制器、响应曲线和参数对比 |
| Phase 3 | 待开始 | 将单关节映射到人体膝关节 | 带重力的膝关节类摆杆 |
| Phase 4 | 待开始 | 进入髋膝踝多关节系统 | 2D 单腿模型与轨迹控制 |
| Phase 5 | 待开始 | 建立外骨骼助力最小闭环 | 膝助力策略及有/无助力对比 |
| Phase 6 | 待开始 | 进入简化人体站立 | 双腿、质心与足部接触分析 |
| Phase 7 | 待开始 | 建立统一的数据终端雏形 | CSV/NumPy 日志与标准分析图 |
| Phase 8 | 待开始 | 理解强化学习环境结构 | Gymnasium 环境，不急于训练策略 |
| Phase 9 | 可选 | 选择深化路线 | 外骨骼控制、机器人迁移或硬件接口 |

## Phase 0：WSL Ubuntu 工程初始化

目标：建立可复现的开发环境和可追溯的工程骨架。

验收标准：

- 仓库位于 WSL home 而不是 `/mnt/c`。
- Git 使用 `main` 分支。
- `scripts/setup_env.sh` 能创建 `.venv` 并安装依赖。
- `scripts/check_env.py` 能验证核心依赖与 MuJoCo 最小模型。
- `pytest` 通过。
- README、环境说明、概念、进展和决策文档齐全。
- GitHub CLI 可用且已登录时关联私有远端；否则保留明确的后续命令。

## Phase 1：单关节模型

目标：理解 `body`、`joint`、`actuator`、自由度、位置、速度和力矩。

实现：创建固定基座、旋转关节、杆件和 motor actuator；用 Python 读取关节状态并施加力矩。

产物：

```text
models/single_joint/single_joint.xml
experiments/001_single_joint_pd/run.py
```

验收：无图形界面也能运行自动化验证，WSLg 环境可通过 `--view` 实时观察动作；打印位置、速度和控制力矩；对应概念写入 `docs/concepts.md`。

完成结果（2026-07-10）：模型包含固定基座、单自由度 hinge、1 kg 杆件和力矩范围 `[-2, 2] N·m` 的 motor；恒力矩 CLI 会按时间打印 `position`、`velocity` 和 `torque`，支持 `--view` GUI 演示，模型结构/运动与 headless CLI 均有自动化测试。

## Phase 2：PD 控制器

目标：通过下面的反馈律控制关节到目标角度：

```text
torque = Kp * (target_position - current_position)
       + Kd * (target_velocity - current_velocity)
```

实现：通用 PD 控制器、状态与力矩记录、角度/速度/力矩曲线，以及不同 `Kp`/`Kd` 的对比。

产物：

```text
src/robo_sim/controllers/pd.py
experiments/001_single_joint_pd/run.py
experiments/001_single_joint_pd/results/
```

## Phase 3：摆杆 / 膝关节类比

目标：模拟带重力的小腿摆动，解释重力矩、维持姿态所需力矩，以及“外骨骼分担人体力矩”的含义。

产物：`models/pendulum/knee_like_pendulum.xml` 和 `experiments/002_pendulum_balance/run.py`。

## Phase 4：二维单腿模型

目标：构建包含 hip、knee、ankle、thigh、shank、foot 的 2D 单腿；分别控制三个关节并记录角度和力矩。

产物：`models/leg_2d/leg_2d.xml` 和 `experiments/003_leg_pd_control/run.py`。

## Phase 5：外骨骼助力模型

目标：用额外 actuator 表示膝关节助力；伸展时提供辅助力矩，屈曲时减少阻碍。

对比指标：human torque、exo torque、total torque、energy proxy、tracking error。

产物：`models/exoskeleton/knee_assist.xml` 和 `experiments/004_knee_assist/run.py`。

## Phase 6：简化人体 / 双腿站立

目标：观察双腿模型的站立、失稳与摔倒，记录质心和足部接触数据。

产物：`models/simple_humanoid/` 和 `experiments/005_standing_balance/`。

## Phase 7：数据记录与分析

目标：统一记录 time、joint position/velocity、torque、contact、body orientation 和 energy proxy，并输出 CSV/NumPy 数据与标准图。

产物：`src/robo_sim/utils/logging.py`、`src/robo_sim/utils/plotting.py` 和实验结果目录。

## Phase 8：强化学习前置准备

目标：把简单控制任务封装为 Gymnasium environment，理解 observation、action、reward、episode 和 policy。

首版 observation 使用关节位置/速度，action 使用力矩，reward 使用跟踪误差与能耗惩罚。此阶段先验证环境，不以训练 PPO 等算法为目标。

## Phase 9：后期可选路线

- 外骨骼控制深化：impedance/admittance control、human-in-the-loop、energy optimization。
- 机器人迁移：motion retargeting、humanoid、policy transfer、Isaac Lab。
- 硬件接口：ROS2、CAN、IMU、电机抽象和实时 logging。

只有前置阶段跑通并记录结果后，才选择其中一条路线。

## Git 工作方式

- `main` 保存可运行、已验证的阶段版本。
- 日常较大改动可放在 `dev`，独立实验可使用 `exp/*`，纯文档可使用 `docs/*`。
- 每个阶段至少一次提交，推荐前缀：`docs:`、`env:`、`sim:`、`ctrl:`、`exp:`。
- 提交前至少运行环境检查和当前测试；生成的大型实验数据不直接提交。
