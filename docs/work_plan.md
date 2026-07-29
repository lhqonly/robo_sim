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
| Phase 2 | 已完成 | 理解反馈与 PD 控制 | 通用 PD 控制器、网页动态响应曲线和在线调参 |
| Phase 2.5 | 已完成 | 理解重力补偿与前馈 | 网页补偿开关、力矩分项和同参数对照实验 |
| Phase 2.6 | 已完成 | 量化控制响应速度和稳定性 | 上升/到达/稳定时间、超调和可调判据 |
| Phase 2.7 | 已完成 | 量化跟踪、动作激烈度和电机负担 | 固定窗口 RMSE、jerk、限幅和机械功 |
| Phase 3 | 已完成 | 将单关节映射到人体膝关节 | 膝关节摆杆、被动下落和重力托举 |
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

验收：无图形界面也能运行自动化验证，WSLg 环境可通过 `--view` 持续观察动作并由用户主动退出；打印位置、速度和控制力矩；对应概念写入 `docs/concepts.md`。

完成结果（2026-07-10）：模型包含固定基座、单自由度 hinge、1 kg 杆件和力矩范围 `[-2, 2] N·m` 的 motor；恒力矩 CLI 会按时间打印 `position`、`velocity` 和 `torque`，支持 3D Viewer + 中文学习控制面板（Watch 下拉、精确扭矩输入、实时状态），模型结构/运动与 headless CLI 均有自动化测试。

## Phase 2：PD 控制器

目标：通过下面的反馈律控制关节到目标角度：

```text
torque = Kp * (target_position - current_position)
       + Kd * (target_velocity - current_velocity)
```

实现：通用 PD 控制器、状态与力矩记录，以及网页实时角度/速度/力矩曲线。

产物：

```text
src/robo_sim/controllers/pd.py
experiments/001_single_joint_pd/run.py
```

完成结果（2026-07-10）：`PDController` 每个物理时间步根据位置/速度误差计算力矩并按执行器范围限幅，支持无窗口数值验证和原生 Viewer。中文学习面板在 PD 模式下可精确修改目标角度、`Kp`、`Kd`，并实时显示误差、P/D 分项、目标保持力矩、当前位置重力/偏置矩与饱和状态；页面直接绘制最近 30 秒的目标/实际角度、速度和 raw/applied 力矩曲线，不生成静态图片。Phase 1 的恒扭矩模式保持兼容。

## Phase 2.5：PD + 重力补偿

目标：在不引入 I 的情况下，先解决纯 PD 必须保留角度误差才能托住杆的问题，并区分“反馈纠偏”和“已知负载补偿”。

控制关系：

```text
总力矩 = P 项 + D 项 + MuJoCo 计算的重力/偏置补偿
实际力矩 = 把总力矩限制在电机的 [-2, 2] N·m 范围内
```

完成结果（2026-07-28）：命令行增加 `--gravity-compensation`，中文面板增加运行时开关；状态卡和动态曲线分别显示 PD、重力补偿和实际总力矩。相同参数目标 `30°`、`Kp=5`、`Kd=3` 下，8 秒纯 PD 停在约 `20.266°`（误差 `9.734°`），开启补偿后到达约 `30.000°`。补偿值直接取自 MuJoCo 对当前模型和姿态计算的 `qfrc_bias`，因此修改杆的质量、长度或姿态后不需要在控制器中重写单杆公式。

## Phase 2.6：阶跃响应时间

目标：不再只凭肉眼判断“快不快、稳不稳”，而是量化从改变目标到关节稳定所需的仿真时间，为比较 `Kp/Kd` 提供统一标准。

默认稳定标准是角度误差进入 `±1°` 并连续保持 `0.5 s`。系统按每个 `0.002 s` MuJoCo 物理步统计首次到达时间、10%～90% 上升时间、稳定时间、确认稳定耗时、最大超调和当前误差；如果稳定后又跑出允许范围，会撤销结果并重新计时。网页允许修改误差范围和连续保持时间，角度曲线同步显示允许范围。

完成结果（2026-07-28）：目标 `45°`、`Kp=5`、`Kd=3`、开启重力补偿时，上升时间约 `1.314 s`，首次进入 `±1°` 和稳定时间约 `2.316 s`，经过 `0.5 s` 持续观察后在 `2.816 s` 确认稳定，最大超调约 `0°`。调参页面将参数、判据、角度曲线和响应指标放在同一区域；一键测试会自动从 `0°` 开始，重复应用相同目标不会覆盖已有测量。

## Phase 2.7：外骨骼调参量化体检

目标：在“快不快、到没到”之外，量化动作激烈程度、总体跟踪误差和执行器负担。

完成结果（2026-07-28）：固定使用目标改变后的前 3 秒，计算角度 RMSE、累计绝对误差 IAE、峰值速度/加速度/jerk、峰值力矩、限幅时间/占比和累计机械功。相同目标、负载和评价窗口下可以公平比较不同 Kp/Kd，同时明确这些是仿真调参指标，不是人体安全阈值。

## Phase 3：摆杆 / 膝关节类比

目标：模拟带重力的小腿摆动，解释重力矩、维持姿态所需力矩，以及“外骨骼分担人体力矩”的含义。

产物：`models/pendulum/knee_like_pendulum.xml` 和 `experiments/002_pendulum_balance/run.py`。

完成结果（2026-07-29）：模型明确包含固定大腿、膝关节、3.0 kg 小腿、0.8 kg 脚和 `±10 N·m` 膝电机。默认从 45° 开始时，MuJoCo 算得保持姿态约需 `6.416 N·m`；被动模式电机输出 0，小腿向自然下垂角落下并摆动，托举模式实时补上重力力矩并保持 45°。中文面板同步显示重力所需总力矩、电机承担、人体概念分担和净力矩。

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
