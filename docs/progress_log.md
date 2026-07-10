# Progress Log

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
- GitHub CLI (`gh`) 尚未安装，因此 Phase 0 先完成本地提交，不自动创建或关联 GitHub 远端。

### 验证

虚拟环境已经创建，所有依赖已安装。当前 Ubuntu 缺少 `python3.12-venv` 的 `ensurepip` 组件，且当前用户没有免密 sudo；安装脚本已验证可使用宿主 pip 的 `--python` 选项，只把 pip 引导进项目 `.venv`，没有向系统 Python 安装项目依赖。

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
