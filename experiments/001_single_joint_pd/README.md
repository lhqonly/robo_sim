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
  --view --torque 1.0 --duration 5
```

Viewer 中蓝色方块是固定基座，橙色杆是运动连杆。拖动鼠标可以旋转视角，滚轮可以缩放；窗口会在仿真时长结束后关闭。

参数：

- `--torque`：恒定电机力矩，单位 N·m，允许范围 `[-2, 2]`。
- `--duration`：仿真时间，单位 s，必须大于 0。
- `--samples`：打印多少个等间隔状态，至少为 2。
- `--view`：打开 GUI 并按真实时间速度运行；不加时采用无窗口快速运行。

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
