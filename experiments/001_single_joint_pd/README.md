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

参数：

- `--torque`：恒定电机力矩，单位 N·m，允许范围 `[-2, 2]`。
- `--duration`：仿真时间，单位 s，必须大于 0。
- `--samples`：打印多少个等间隔状态，至少为 2。

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

本阶段不打开 viewer，也不绘图。先确认数值状态和物理方向正确，Phase 2 再加入 PD 控制与响应曲线。
