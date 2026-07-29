# Experiments

Experiments are numbered in learning order and should be independently runnable. Each experiment must explain its question, parameters, run command, expected result and generated files. Generated plots/data go into a local `results/` directory and are ignored unless a small reference result is intentionally committed.

当前实验：

- `001_single_joint_pd/`：Phase 1 使用恒定力矩观察单关节运动；Phase 2 使用 PD 反馈控制目标角度并绘制响应曲线。
- `002_pendulum_balance/`：Phase 3 把单杆映射为膝关节、小腿和脚，对比被动下落与重力托举。
