# WSL2 Ubuntu 环境指南

## 1. 为什么放在 WSL home

项目路径使用：

```text
/home/lhq24/robo_sim
```

也就是 shell 中的 `~/robo_sim`。不要放在 `/mnt/c`：WSL 直接访问 Linux 文件系统时，通常有更好的大量小文件 I/O 性能；Python 虚拟环境、文件权限、符号链接和 Git 行为也更接近生产 Linux。

Windows 编辑器仍可通过 WSL 集成打开该目录，例如在项目目录运行 `code .`（需安装 VS Code WSL 扩展）。

## 2. 系统前置条件

检查现有工具：

```bash
python3 --version
git --version
```

本项目要求 Python 3.10+。2026-07-10 当前机器检测到 Python 3.12.3，满足要求，不需要为了版本号完全相同而降级。

当前 WSL 还通过 ROS 2 暴露了 `launch_testing` pytest 插件，因此项目暂时约束 pytest `<9`；pytest 9 已移除该插件仍使用的兼容接口。

如果新 Ubuntu 缺少 venv/pip，可安装：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

## 3. 创建并激活项目环境

推荐使用项目脚本：

```bash
cd ~/robo_sim
bash scripts/setup_env.sh
source .venv/bin/activate
```

等价的手工过程是：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

虚拟环境只需创建一次；新开终端后重新执行 `source .venv/bin/activate` 即可。

## 4. 验证

```bash
python scripts/check_env.py
pytest
```

检查脚本不会打开 MuJoCo viewer，因此可在没有桌面/显卡配置的 WSL 终端中运行。后续需要图形窗口时，再单独验证 WSLg/OpenGL。

## 5. GitHub 远端

当前机器未安装 `gh`，所以没有自动创建远端。安装 GitHub CLI 后登录：

```bash
gh auth login
gh auth status
```

登录时选择 GitHub.com、HTTPS 和浏览器认证。确认本地提交无误后，在项目目录创建私有仓库并推送：

```bash
gh repo create lhqonly/robo_sim --private --source=. --remote=origin --push
```

如果你先在 GitHub 网页创建了空仓库，不要勾选自动生成 README，然后关联：

```bash
git remote add origin https://github.com/lhqonly/robo_sim.git
git push -u origin main
```

## 6. 常见问题

- `No module named ...`：确认已激活 `.venv`，并用 `which python` 检查是否指向项目目录。
- `ensurepip is not available`：安装 `python3-venv` 后删除未完成的 `.venv`，再运行 setup 脚本。
- MuJoCo viewer 无法打开：Phase 0 的检查不依赖 viewer；后续先确认 Windows 版本支持 WSLg，并更新 WSL/显卡驱动。
- 不要使用 `sudo pip install`：它会修改系统 Python，破坏项目隔离。
