"""Local Chinese learning panel for the single-joint experiments."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mujoco

from robo_sim.controllers.pd import PDController


PANEL_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LinkJoin 单关节学习面板</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, "Microsoft YaHei", sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 780px; margin: auto; padding: 24px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .muted { color: #9ca3af; }
    .card { margin-top: 18px; padding: 18px; background: #1f2937;
            border: 1px solid #374151; border-radius: 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .metric { padding: 12px; background: #111827; border-radius: 8px; }
    .metric strong { display: block; color: #93c5fd; font-size: 20px; margin-top: 5px; }
    label { display: block; margin-bottom: 7px; font-weight: 650; }
    input, select, button { box-sizing: border-box; border: 1px solid #4b5563;
      border-radius: 7px; background: #111827; color: #f9fafb; padding: 9px 11px; }
    input[type=number], select { width: 100%; font-size: 16px; }
    input[type=range] { width: 100%; padding: 0; accent-color: #f59e0b; }
    button { cursor: pointer; margin: 6px 5px 0 0; }
    button:hover { background: #374151; }
    .primary { background: #2563eb; border-color: #3b82f6; }
    .danger { background: #7f1d1d; border-color: #b91c1c; }
    .watch-value { color: #fbbf24; font-size: 34px; font-variant-numeric: tabular-nums; }
    .description { min-height: 44px; margin-top: 10px; color: #d1d5db; }
    .status { margin-top: 10px; color: #86efac; min-height: 22px; }
    code { color: #fde68a; }
    @media (max-width: 580px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>LinkJoin 单关节学习面板</h1>
  <div class="muted">这是 MuJoCo Viewer 的中文辅助面板；只监听本机 127.0.0.1。</div>
  <div id="modeBadge" class="status">正在读取控制模式……</div>

  <section id="torqueControl" class="card">
    <label for="watchField">Watch Field（观察变量）</label>
    <select id="watchField">
      <option value="qpos">qpos — 关节角度</option>
      <option value="qvel">qvel — 关节角速度</option>
      <option value="ctrl">ctrl — 电机指令扭矩</option>
      <option value="actuator_force">actuator_force — 执行器实际输出</option>
    </select>
    <div id="watchValue" class="watch-value">--</div>
    <div id="watchDescription" class="description"></div>
  </section>

  <section id="pdControl" class="card" hidden>
    <h2 style="margin-top:0">PD 闭环参数</h2>
    <div class="grid">
      <div>
        <label for="targetInput">目标角度（degree / °）</label>
        <input id="targetInput" type="number" min="-120" max="120" step="0.1" />
      </div>
      <div>
        <label for="kpInput">比例增益 Kp（N·m/rad）</label>
        <input id="kpInput" type="number" min="0" max="200" step="0.1" />
      </div>
      <div>
        <label for="kdInput">微分增益 Kd（N·m·s/rad）</label>
        <input id="kdInput" type="number" min="0" max="50" step="0.1" />
      </div>
    </div>
    <button id="applyPd" class="primary">应用目标与增益</button>
    <div class="muted">Kp 像弹簧，把关节拉向目标；Kd 像阻尼，抑制速度和振荡。</div>
    <div id="pdStatus" class="status"></div>
  </section>

  <section class="card">
    <label for="torqueInput">joint_motor 精确扭矩输入（N·m）</label>
    <input id="torqueInput" type="number" min="-2" max="2" step="0.01" value="0" />
    <button id="applyTorque" class="primary">应用精确值</button>
    <button data-torque="-2">-2</button><button data-torque="-1">-1</button>
    <button data-torque="-0.5">-0.5</button><button data-torque="0">0</button>
    <button data-torque="0.5">+0.5</button><button data-torque="1">+1</button>
    <button data-torque="2">+2</button>
    <label for="torqueSlider" style="margin-top:14px">快速拖动（-2 ～ +2 N·m）</label>
    <input id="torqueSlider" type="range" min="-2" max="2" step="0.01" value="0" />
    <div class="muted">数字输入用于精确实验；滑块用于快速观察。两者与 Viewer 的紫色 Control 同步。</div>
    <div id="status" class="status"></div>
  </section>

  <section class="card">
    <div class="grid">
      <div class="metric">仿真时间 time<strong id="time">--</strong></div>
      <div class="metric">关节角度 qpos<strong id="qpos">--</strong></div>
      <div class="metric">角速度 qvel<strong id="qvel">--</strong></div>
      <div class="metric">电机输入 ctrl<strong id="ctrl">--</strong></div>
      <div id="errorMetric" class="metric" hidden>位置误差<strong id="positionError">--</strong></div>
      <div id="rawTorqueMetric" class="metric" hidden>限幅前 PD 力矩<strong id="rawTorque">--</strong></div>
    </div>
    <button id="reset" class="danger">重置姿态</button>
  </section>

  <section class="card muted">
    <strong style="color:#e5e7eb">最重要的关系</strong><br/>
    <code id="controlRelation">ctrl（输入扭矩） → MuJoCo 动力学 → qpos/qvel（角度与速度）</code><br/>
    关闭 3D Viewer 或在启动它的终端按 Ctrl+C，会同时关闭本面板服务。
  </section>
</main>
<script>
const definitions = {
  qpos: {unit: 'rad', text: '关节当前位置。弧度换算角度：rad × 180 / π。'},
  qvel: {unit: 'rad/s', text: '关节角速度。正负号表示方向，接近 0 表示基本停止。'},
  ctrl: {unit: 'N·m', text: '发送给 joint_motor 的指令扭矩，是输入，不是目标角度。'},
  actuator_force: {unit: 'N·m', text: 'MuJoCo 执行器当前实际输出；本模型 gear=1，通常接近 ctrl。'}
};
let latest = null;
const $ = (id) => document.getElementById(id);

function format(value, digits=6) { return Number(value).toFixed(digits); }
function renderWatch() {
  if (!latest) return;
  const key = $('watchField').value;
  const def = definitions[key];
  $('watchValue').textContent = `${format(latest[key])} ${def.unit}`;
  $('watchDescription').textContent = def.text;
}
function renderState(state) {
  latest = state;
  const pdMode = state.mode === 'pd';
  $('modeBadge').textContent = pdMode
    ? 'Phase 2：PD 闭环控制（根据误差实时改变扭矩）'
    : 'Phase 1：恒扭矩开环控制';
  $('torqueControl').hidden = pdMode;
  $('pdControl').hidden = !pdMode;
  $('errorMetric').hidden = !pdMode;
  $('rawTorqueMetric').hidden = !pdMode;
  $('time').textContent = `${format(state.time, 3)} s`;
  $('qpos').textContent = `${format(state.qpos)} rad / ${format(state.qpos_deg, 2)}°`;
  $('qvel').textContent = `${format(state.qvel)} rad/s`;
  $('ctrl').textContent = `${format(state.ctrl, 3)} N·m`;
  if (pdMode) {
    $('positionError').textContent = `${format(state.position_error_rad)} rad / ${format(state.position_error_deg, 2)}°`;
    $('rawTorque').textContent = `${format(state.raw_torque_nm, 3)} N·m`;
    $('pdStatus').textContent = state.saturated
      ? '当前已触及 ±2 N·m 力矩限幅（橙色杆不会得到更大的力矩）'
      : '当前力矩未触及限幅';
    if (document.activeElement !== $('targetInput')) $('targetInput').value = format(state.target_position_deg, 2);
    if (document.activeElement !== $('kpInput')) $('kpInput').value = format(state.kp, 2);
    if (document.activeElement !== $('kdInput')) $('kdInput').value = format(state.kd, 2);
    $('controlRelation').textContent = '目标角度 − qpos → PD 控制器 → ctrl → MuJoCo → 新的 qpos/qvel';
    definitions.ctrl.text = 'PD 控制器根据角度/速度误差实时算出的电机扭矩；原生 Viewer 紫色滑块会被控制器持续更新。';
  } else {
    if (document.activeElement !== $('torqueInput')) $('torqueInput').value = format(state.ctrl, 3);
    if (document.activeElement !== $('torqueSlider')) $('torqueSlider').value = state.ctrl;
  }
  renderWatch();
}
async function refresh() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    renderState(await response.json());
  } catch (_) {
    $('status').textContent = '等待 Viewer 数据……';
  }
}
async function post(path, body={}) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '操作失败');
  return result;
}
async function setTorque(value) {
  try {
    const result = await post('/api/torque', {value: Number(value)});
    $('status').textContent = `已应用 ${format(result.ctrl, 3)} N·m`;
    await refresh();
  } catch (error) { $('status').textContent = error.message; }
}
async function setPd() {
  try {
    const result = await post('/api/pd', {
      target_deg: Number($('targetInput').value),
      kp: Number($('kpInput').value),
      kd: Number($('kdInput').value)
    });
    $('pdStatus').textContent = `已应用：目标 ${format(result.target_position_deg, 2)}°，Kp=${format(result.kp, 2)}，Kd=${format(result.kd, 2)}`;
    await refresh();
  } catch (error) { $('pdStatus').textContent = error.message; }
}
$('watchField').addEventListener('change', renderWatch);
$('applyTorque').addEventListener('click', () => setTorque($('torqueInput').value));
$('torqueInput').addEventListener('keydown', (event) => { if (event.key === 'Enter') setTorque(event.target.value); });
$('torqueSlider').addEventListener('change', (event) => setTorque(event.target.value));
document.querySelectorAll('[data-torque]').forEach((button) => button.addEventListener('click', () => setTorque(button.dataset.torque)));
$('applyPd').addEventListener('click', setPd);
[$('targetInput'), $('kpInput'), $('kdInput')].forEach((input) => input.addEventListener('keydown', (event) => { if (event.key === 'Enter') setPd(); }));
$('reset').addEventListener('click', async () => { await post('/api/reset'); $('status').textContent = '已重置'; await refresh(); });
refresh();
setInterval(refresh, 120);
</script>
</body>
</html>
"""


class LearningPanelServer:
    """Serve a localhost-only panel backed by live ``mjData`` values."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_id: int,
        actuator_id: int,
        port: int = 0,
        pd_controller: PDController | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.joint_id = joint_id
        self.actuator_id = actuator_id
        self.pd_controller = pd_controller
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", port), self._make_handler_type()
        )
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="robo-sim-learning-panel",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def snapshot(self) -> dict[str, float | str | bool]:
        qpos_address = self.model.jnt_qposadr[self.joint_id]
        qvel_address = self.model.jnt_dofadr[self.joint_id]
        with self._lock:
            qpos = float(self.data.qpos[qpos_address])
            snapshot: dict[str, float | str | bool] = {
                "time": float(self.data.time),
                "qpos": qpos,
                "qpos_deg": qpos * 180.0 / 3.141592653589793,
                "qvel": float(self.data.qvel[qvel_address]),
                "ctrl": float(self.data.ctrl[self.actuator_id]),
                "actuator_force": float(
                    self.data.actuator_force[self.actuator_id]
                ),
                "mode": "pd" if self.pd_controller is not None else "torque",
            }
        if self.pd_controller is not None:
            output = self.pd_controller.compute(
                position_rad=float(snapshot["qpos"]),
                velocity_rad_s=float(snapshot["qvel"]),
            )
            settings = self.pd_controller.settings()
            target_rad = settings["target_position_rad"]
            snapshot.update(
                {
                    "kp": settings["kp"],
                    "kd": settings["kd"],
                    "target_position_rad": target_rad,
                    "target_position_deg": target_rad * 180.0 / 3.141592653589793,
                    "position_error_rad": output.position_error_rad,
                    "position_error_deg": output.position_error_rad
                    * 180.0
                    / 3.141592653589793,
                    "raw_torque_nm": output.raw_torque_nm,
                    "saturated": output.saturated,
                }
            )
        return snapshot

    def set_torque(self, value: float) -> float:
        if self.pd_controller is not None:
            raise ValueError("PD 模式下扭矩由控制器计算，请修改目标角度、Kp 或 Kd")
        control_min, control_max = self.model.actuator_ctrlrange[self.actuator_id]
        if not control_min <= value <= control_max:
            raise ValueError(
                f"扭矩必须在 [{control_min:g}, {control_max:g}] N·m 范围内"
            )
        with self._lock:
            self.data.ctrl[self.actuator_id] = value
        return value

    def set_pd(self, target_deg: float, kp: float, kd: float) -> dict[str, float]:
        if self.pd_controller is None:
            raise ValueError("当前不是 PD 控制模式")
        target_rad = target_deg * 3.141592653589793 / 180.0
        joint_min, joint_max = self.model.jnt_range[self.joint_id]
        if not joint_min <= target_rad <= joint_max:
            raise ValueError(
                f"目标角度必须在 {joint_min * 180 / 3.141592653589793:.1f}° "
                f"到 {joint_max * 180 / 3.141592653589793:.1f}° 之间"
            )
        if kp > 200:
            raise ValueError("Kp 不能大于 200")
        if kd > 50:
            raise ValueError("Kd 不能大于 50")
        self.pd_controller.update(
            kp=kp, kd=kd, target_position_rad=target_rad
        )
        settings = self.pd_controller.settings()
        return {
            "target_position_rad": settings["target_position_rad"],
            "target_position_deg": settings["target_position_rad"]
            * 180.0
            / 3.141592653589793,
            "kp": settings["kp"],
            "kd": settings["kd"],
        }

    def reset(self) -> None:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.ctrl[self.actuator_id] = 0.0

    def _make_handler_type(self) -> type[BaseHTTPRequestHandler]:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
                if self.path == "/" or self.path.startswith("/?"):
                    self._send_bytes(
                        PANEL_HTML.encode("utf-8"), "text/html; charset=utf-8"
                    )
                elif self.path == "/api/state":
                    self._send_json(panel.snapshot())
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
                try:
                    body = self._read_json()
                    if self.path == "/api/torque":
                        value = panel.set_torque(float(body["value"]))
                        self._send_json({"ctrl": value})
                    elif self.path == "/api/pd":
                        self._send_json(
                            panel.set_pd(
                                float(body["target_deg"]),
                                float(body["kp"]),
                                float(body["kd"]),
                            )
                        )
                    elif self.path == "/api/reset":
                        panel.reset()
                        self._send_json({"ok": True})
                    else:
                        self._send_json(
                            {"error": "not found"}, HTTPStatus.NOT_FOUND
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json(
                        {"error": str(exc) or "无效请求"}, HTTPStatus.BAD_REQUEST
                    )

            def _read_json(self) -> dict[str, Any]:
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                return json.loads(self.rfile.read(length) or b"{}")

            def _send_json(
                self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
            ) -> None:
                self._send_bytes(
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status,
                )

            def _send_bytes(
                self,
                payload: bytes,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    # A readiness probe may close its short-lived connection
                    # before the response reaches the socket.
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler
