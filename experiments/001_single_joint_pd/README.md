# Experiment 001：单关节恒力矩

## 目标

验证机器人/外骨骼最小链路：Python 写入 motor 力矩，MuJoCo 推进动力学，再由 Python 读取关节位置与速度。

当前是 Phase 1 的开环实验，目录名中的 PD 会在 Phase 2 实现。

## 模型

- 固定基座
- 一个绕 Y 轴旋转的 hinge joint
- 0.5 m、1 kg 的杆件
- 力矩范围 `[-2, 2] N·m` 的 motor actuator
- 仿真时间步长 `0.002 s`，包含重力和关节阻尼

## 运行

```bash
cd ~/robo_sim
source .venv/bin/activate
python experiments/001_single_joint_pd/run.py \
  --torque 0.5 --duration 1.0 --samples 6
```

当前 WSL 支持 WSLg 时，可打开 MuJoCo Viewer 实时观察：

```bash
python experiments/001_single_joint_pd/run.py \
  --view --torque 1.0
```

Viewer 中蓝色方块是固定基座，黄色圆柱是简化的关节/电机位置，橙色杆是运动连杆。拖动鼠标可以旋转视角，滚轮可以缩放。GUI 会一直运行，关闭 Viewer 窗口或先点击终端再按 `Ctrl+C` 才退出。GUI 在隔离的子进程中运行，因此其原生窗口清理不会占住终端。

同时会自动打开一个浏览器中文学习面板。它只监听本机 `127.0.0.1`，关闭 Viewer 或按 `Ctrl+C` 后服务随即停止。

### 中文学习面板

原生 Viewer 的 `Watch.Field` 是自由文本，原生 `joint_motor` 也是只有滑块的固定控件；Python Viewer API 不能直接把它们替换为自定义控件。因此项目增加了一个与同一份 MuJoCo `data` 实时同步的辅助面板：

- Watch 下拉框：`qpos`（关节角度）、`qvel`（角速度）、`ctrl`（指令扭矩）、`actuator_force`（实际输出）。
- 每个字段显示中文含义、单位和实时值。
- 精确扭矩输入框：可键入例如 `1.234`，按 Enter 或点击“应用精确值”。
- 快速滑块与 `-2/-1/-0.5/0/+0.5/+1/+2` 预设按钮。
- 重置按钮：恢复初始姿态并清零扭矩。

原生 Viewer 的紫色 Control 和中文面板控制的是同一个 `data.ctrl[0]`，任一侧修改都会在另一侧体现。

如果浏览器没有自动打开，复制终端输出，例如：

```text
Chinese learning panel: http://127.0.0.1:53017/
```

也可以禁止自动打开浏览器：

```bash
python experiments/001_single_joint_pd/run.py \
  --view --torque 1.0 --no-browser
```

参数：

- `--torque`：恒定电机力矩，单位 N·m，允许范围 `[-2, 2]`。
- `--duration`：无窗口模式的仿真时间，单位 s，必须大于 0。
- `--samples`：无窗口模式打印多少个等间隔状态，至少为 2。
- `--view`：打开交互式 GUI，忽略 `--duration/--samples`，直到人工退出；不加时采用无窗口快速运行。
- `--no-browser`：仍启动中文学习面板服务，但不自动打开浏览器。

## 为什么做这个实验

它验证的不是“外骨骼已经能帮助人走路”，而是后续所有控制的最小基础是否正确：

1. XML 中的固定端、活动杆和旋转轴是否装配正确。
2. Python 发出的正/负电机力矩是否进入了预期关节。
3. MuJoCo 计算出的角度和速度方向是否符合直觉。
4. 重力、惯性和阻尼能否共同产生稳定、可解释的运动。

以 `1 N·m` 为例，电机始终输出同一个力矩。它不是命令关节“转到 1 rad”。杆长 0.5 m、质量 1 kg，质心距关节约 0.25 m；静止时满足：

```text
电机力矩 = 重力力矩
1 ≈ 1 × 9.81 × 0.25 × sin(angle)
angle ≈ 0.420 rad ≈ 24.1°
```

所以杆件先转动，阻尼逐渐消耗速度，最后停在约 24°。终端中接近 `0.419982 rad` 的结果正好验证了这个力矩平衡。

本实验还不能验证真实膝关节精度、人体舒适性、硬件电机性能或外骨骼助力效果；这些需要后续模型和真实数据。

## 如何读结果

输出列分别是时间、关节角度、角速度和力矩。默认正力矩会让角度和速度先变成正值。随着杆件抬起，重力恢复力矩与阻尼开始抵消电机力矩，所以速度不必一直增加。

可以对比：

```bash
python experiments/001_single_joint_pd/run.py --torque 0
python experiments/001_single_joint_pd/run.py --torque 0.5
python experiments/001_single_joint_pd/run.py --torque -0.5
```

零力矩时，初始杆件位于竖直最低点，应基本保持不动；负力矩会产生相反方向的运动。

## 验证

```bash
python -m pytest tests/test_single_joint.py
```

自动化测试使用无窗口模式，保证可重复并适合 CI；人工验收再使用 `--view` 观察动作。Phase 2 将加入 PD 控制与响应曲线。

如果 Viewer 异常退出后终端把 `Ctrl+C` 显示成 `^C`，输入 `stty sane` 并按 Enter，或在 VS Code 中关闭该终端并新建一个。
