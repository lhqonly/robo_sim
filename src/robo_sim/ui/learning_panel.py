"""Local Chinese learning panel for the single-joint experiments."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mujoco

from robo_sim.controllers.gravity import GravityCompensationSwitch
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
    main { max-width: 980px; margin: auto; padding: 24px; }
    [hidden] { display: none !important; }
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
    .insight { line-height: 1.7; padding: 12px; background: #111827;
               border-left: 4px solid #f59e0b; border-radius: 6px; }
    .chart-block { margin-top: 16px; padding: 12px; background: #111827;
                   border-radius: 8px; }
    .chart-title { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
                   margin-bottom: 8px; font-weight: 650; }
    .legend { font-size: 13px; font-weight: 400; color: #d1d5db; }
    .legend::before { content: ''; display: inline-block; width: 18px;
                      border-top: 3px solid var(--legend-color); margin-right: 5px;
                      vertical-align: middle; }
    canvas { display: block; width: 100%; height: 220px; }
    code { color: #fde68a; }
    @media (max-width: 580px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>LinkJoin 单关节学习面板</h1>
  <div class="muted">这是 MuJoCo Viewer 的中文辅助面板；只监听本机 127.0.0.1。</div>
  <div id="modeBadge" class="status">正在读取控制模式……</div>

  <section id="watchControl" class="card">
    <label for="watchField">Watch Field（观察变量）</label>
    <select id="watchField">
      <option value="qpos">qpos — 关节角度（rad / degree）</option>
      <option value="qvel">qvel — 关节角速度（rad/s / degree/s）</option>
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
    <button id="toggleGravityCompensation">重力补偿：关闭（点击开启）</button>
    <div class="muted">PD 是 PID 里的 P + D：P 看“还差多远”，D 看“现在动得多快”并负责刹车。I 会累积长期误差，后续单独学习。</div>
    <div class="muted">重力补偿不是 I：它让模型先用一部分力托住杆的重量，再由 PD 负责对准目标和刹车。</div>
    <div id="pdStatus" class="status"></div>
  </section>

  <section id="torqueControl" class="card" hidden>
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
      <div id="rawTorqueMetric" class="metric" hidden>限幅前总力矩<strong id="rawTorque">--</strong></div>
      <div id="pdTorqueMetric" class="metric" hidden>PD 力矩（P + D）<strong id="pdTorque">--</strong></div>
      <div id="gravityTorqueMetric" class="metric" hidden>重力补偿力矩<strong id="gravityTorque">--</strong></div>
    </div>
    <button id="reset" class="danger">重置姿态</button>
  </section>

  <section id="pdInsightCard" class="card" hidden>
    <h2 id="equilibriumTitle" style="margin-top:0">为什么停在这里，没有到目标？</h2>
    <div id="equilibriumExplanation" class="insight">正在计算重力与 PD 平衡……</div>
    <div class="grid" style="margin-top:12px">
      <div class="metric">停在目标需要的力矩<strong id="targetHoldTorque">--</strong></div>
      <div class="metric">停在当前位置需要的力矩<strong id="biasTorque">--</strong></div>
      <div class="metric">P 项：Kp × 位置误差<strong id="pTorque">--</strong></div>
      <div class="metric">D 项：Kd × 速度误差<strong id="dTorque">--</strong></div>
      <div class="metric">重力补偿项<strong id="gravityInsightTorque">--</strong></div>
    </div>
  </section>

  <section id="responseCard" class="card" hidden>
    <h2 style="margin:0">实时响应曲线</h2>
    <div class="muted">记录最近 30 秒。修改目标时，绿色虚线会形成阶跃，可直接观察实际角度如何追踪。</div>
    <div id="curveStatus" class="status">正在等待仿真数据……</div>
    <button id="toggleRecording">暂停记录</button>
    <button id="clearCharts">清空曲线</button>

    <div class="chart-block">
      <div class="chart-title">角度响应
        <span class="legend" style="--legend-color:#34d399">目标角度</span>
        <span class="legend" style="--legend-color:#60a5fa">实际角度</span>
      </div>
      <canvas id="angleChart"></canvas>
    </div>
    <div class="chart-block">
      <div class="chart-title">角速度
        <span class="legend" style="--legend-color:#fbbf24">实际角速度</span>
      </div>
      <canvas id="velocityChart"></canvas>
    </div>
    <div class="chart-block">
      <div class="chart-title">控制力矩
        <span class="legend" style="--legend-color:#a78bfa">PD（P + D）</span>
        <span class="legend" style="--legend-color:#34d399">重力补偿</span>
        <span class="legend" style="--legend-color:#f97316">实际总力矩</span>
        <span class="legend" style="--legend-color:#ef4444">±2 N·m 限幅</span>
      </div>
      <canvas id="torqueChart"></canvas>
    </div>
  </section>

  <section class="card muted">
    <strong style="color:#e5e7eb">最重要的关系</strong><br/>
    <code id="controlRelation">ctrl（输入扭矩） → MuJoCo 动力学 → qpos/qvel（角度与速度）</code><br/>
    关闭 3D Viewer 或在启动它的终端按 Ctrl+C，会同时关闭本面板服务。
  </section>
</main>
<script>
const definitions = {
  qpos: {unit: 'rad', text: '关节当前位置，同时显示弧度 rad 和角度 degree。'},
  qvel: {unit: 'rad/s', text: '关节角速度，同时显示 rad/s 和 degree/s；接近 0 表示基本停止。'},
  ctrl: {unit: 'N·m', text: '发送给 joint_motor 的指令扭矩，是输入，不是目标角度。'},
  actuator_force: {unit: 'N·m', text: 'MuJoCo 执行器当前实际输出；本模型 gear=1，通常接近 ctrl。'}
};
let latest = null;
let responseHistory = [];
let recording = true;
let lastRecordedTime = null;
let lastAdvanceWallTime = Date.now();
const $ = (id) => document.getElementById(id);

function format(value, digits=6) { return Number(value).toFixed(digits); }
function renderWatch() {
  if (!latest) return;
  const key = $('watchField').value;
  const def = definitions[key];
  if (key === 'qpos') {
    $('watchValue').textContent = `${format(latest.qpos)} rad / ${format(latest.qpos_deg, 2)}°`;
  } else if (key === 'qvel') {
    $('watchValue').textContent = `${format(latest.qvel)} rad/s / ${format(latest.qvel * 180 / Math.PI, 2)}°/s`;
  } else {
    $('watchValue').textContent = `${format(latest[key])} ${def.unit}`;
  }
  $('watchDescription').textContent = def.text;
}
function renderState(state) {
  latest = state;
  const pdMode = state.mode === 'pd';
  $('modeBadge').textContent = pdMode
    ? (state.gravity_compensation_enabled
      ? 'Phase 2.5：PD + 重力补偿（先托住重量，再纠正误差）'
      : 'Phase 2：纯 PD 闭环控制（根据误差实时改变扭矩）')
    : 'Phase 1：恒扭矩开环控制';
  $('torqueControl').hidden = pdMode;
  $('pdControl').hidden = !pdMode;
  $('errorMetric').hidden = !pdMode;
  $('rawTorqueMetric').hidden = !pdMode;
  $('pdTorqueMetric').hidden = !pdMode;
  $('gravityTorqueMetric').hidden = !pdMode;
  $('pdInsightCard').hidden = !pdMode;
  $('responseCard').hidden = !pdMode;
  $('time').textContent = `${format(state.time, 3)} s`;
  $('qpos').textContent = `${format(state.qpos)} rad / ${format(state.qpos_deg, 2)}°`;
  $('qvel').textContent = `${format(state.qvel)} rad/s`;
  $('ctrl').textContent = `${format(state.ctrl, 3)} N·m`;
  if (pdMode) {
    $('positionError').textContent = `${format(state.position_error_rad)} rad / ${format(state.position_error_deg, 2)}°`;
    $('rawTorque').textContent = `${format(state.raw_torque_nm, 3)} N·m`;
    $('pdTorque').textContent = `${format(state.pd_torque_nm, 3)} N·m`;
    $('gravityTorque').textContent = `${format(state.gravity_compensation_torque_nm, 3)} N·m`;
    $('targetHoldTorque').textContent = `${format(state.target_hold_torque_nm, 3)} N·m`;
    $('biasTorque').textContent = `${format(state.bias_torque_nm, 3)} N·m`;
    $('pTorque').textContent = `${format(state.proportional_torque_nm, 3)} N·m`;
    $('dTorque').textContent = `${format(state.derivative_torque_nm, 3)} N·m`;
    $('gravityInsightTorque').textContent = `${format(state.gravity_compensation_torque_nm, 3)} N·m`;
    $('toggleGravityCompensation').textContent = state.gravity_compensation_enabled
      ? '重力补偿：已开启（点击关闭）'
      : '重力补偿：关闭（点击开启）';
    $('toggleGravityCompensation').className = state.gravity_compensation_enabled ? 'primary' : '';
    $('pdStatus').textContent = state.saturated
      ? '当前已触及 ±2 N·m 力矩限幅（橙色杆不会得到更大的力矩）'
      : '当前力矩未触及限幅';
    if (document.activeElement !== $('targetInput')) $('targetInput').value = format(state.target_position_deg, 2);
    if (document.activeElement !== $('kpInput')) $('kpInput').value = format(state.kp, 2);
    if (document.activeElement !== $('kdInput')) $('kdInput').value = format(state.kd, 2);
    $('controlRelation').textContent = state.gravity_compensation_enabled
      ? '重力补偿先托住重量 + PD 修正误差 → ctrl → MuJoCo → 新的 qpos/qvel'
      : '目标角度 − qpos → PD 控制器 → ctrl → MuJoCo → 新的 qpos/qvel';
    definitions.ctrl.text = state.gravity_compensation_enabled
      ? 'PD 力矩与重力补偿相加、再经过限幅后的电机总扭矩；原生 Viewer 紫色滑块会被持续更新。'
      : '纯 PD 根据角度/速度误差实时算出的电机扭矩；原生 Viewer 紫色滑块会被持续更新。';
    const withinLimit = Math.abs(state.target_hold_torque_nm) <= 2;
    const torquePerDegree = state.kp * Math.PI / 180;
    if (state.gravity_compensation_enabled) {
      $('equilibriumTitle').textContent = '重力补偿现在做了什么？';
      $('equilibriumExplanation').textContent =
        `MuJoCo 根据杆的质量和当前姿态，算出现在需要约 ${format(state.bias_torque_nm, 3)} N·m 才能托住重量。` +
        `重力补偿先给出 ${format(state.gravity_compensation_torque_nm, 3)} N·m；PD 不再需要故意保留很大的角度误差来出力，` +
        `只负责修正剩余的 ${format(Math.abs(state.position_error_deg), 2)}° 误差和刹车。` +
        `两部分相加后再经过 ±2 N·m 电机限幅。`;
    } else {
      $('equilibriumTitle').textContent = '为什么停在这里，没有到目标？';
      $('equilibriumExplanation').textContent =
        `想停在 ${format(state.target_position_deg, 1)}°，电机需要持续提供约 ${format(state.target_hold_torque_nm, 3)} N·m，` +
        `${withinLimit ? '这个数没有超过电机上限。' : '这个数已经超过电机上限。'}` +
        `但纯 PD 一旦完全到达目标，误差就变成 0，它反而会命令电机输出 0，杆就会被重力拉回来。` +
        `现在 Kp=${format(state.kp, 1)}，每相差 1° 只能增加约 ${format(torquePerDegree, 3)} N·m。` +
        `小误差产生的力气不够，所以必须退到相差 ${format(Math.abs(state.position_error_deg), 2)}°，` +
        `此时 P 项达到 ${format(state.proportional_torque_nm, 3)} N·m，刚好托住当前位置，杆才停下。`;
    }
    appendResponsePoint(state);
  } else {
    if (document.activeElement !== $('torqueInput')) $('torqueInput').value = format(state.ctrl, 3);
    if (document.activeElement !== $('torqueSlider')) $('torqueSlider').value = state.ctrl;
  }
  renderWatch();
}

function clearResponseHistory() {
  responseHistory = [];
  lastRecordedTime = null;
  drawAllCharts();
}

function appendResponsePoint(state) {
  if (!recording) return;
  if (lastRecordedTime !== null && state.time < lastRecordedTime) clearResponseHistory();
  if (lastRecordedTime !== null && Math.abs(state.time - lastRecordedTime) < 1e-6) {
    if (Date.now() - lastAdvanceWallTime > 1000) {
      $('curveStatus').textContent = '仿真现在是暂停状态：请在 MuJoCo Viewer 左侧 Simulation 区域点击 Run。';
    }
    return;
  }
  lastAdvanceWallTime = Date.now();
  $('curveStatus').textContent = '正在记录动态曲线';
  lastRecordedTime = state.time;
  responseHistory.push({
    time: state.time,
    target: state.target_position_deg,
    position: state.qpos_deg,
    velocity: state.qvel * 180 / Math.PI,
    pdTorque: state.pd_torque_nm,
    gravityTorque: state.gravity_compensation_torque_nm,
    torque: state.ctrl
  });
  const cutoff = state.time - 30;
  while (responseHistory.length > 2 && responseHistory[0].time < cutoff) responseHistory.shift();
  drawAllCharts();
}

function drawChart(canvas, series, options={}) {
  const width = Math.max(canvas.clientWidth, 320);
  const height = 220;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const margin = {left: 58, right: 14, top: 12, bottom: 30};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.fillStyle = '#9ca3af';
  context.font = '12px system-ui, sans-serif';
  if (responseHistory.length < 2) {
    context.fillText('等待至少两个实时采样点……', margin.left, margin.top + 24);
    return;
  }

  const xMin = responseHistory[0].time;
  const xMax = Math.max(responseHistory[responseHistory.length - 1].time, xMin + 0.1);
  const values = [];
  for (const point of responseHistory) {
    for (const item of series) values.push(item.value(point));
  }
  if (options.extraValues) values.push(...options.extraValues);
  if (options.includeZero) values.push(0);
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  const padding = Math.max((yMax - yMin) * 0.12, options.minimumPadding || 0.5);
  yMin -= padding;
  yMax += padding;
  const toX = (value) => margin.left + (value - xMin) / (xMax - xMin) * plotWidth;
  const toY = (value) => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;

  context.strokeStyle = '#374151';
  context.lineWidth = 1;
  context.fillStyle = '#9ca3af';
  for (let index = 0; index <= 4; index += 1) {
    const fraction = index / 4;
    const y = margin.top + fraction * plotHeight;
    const value = yMax - fraction * (yMax - yMin);
    context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
    context.fillText(value.toFixed(options.decimals ?? 1), 5, y + 4);
    const x = margin.left + fraction * plotWidth;
    const time = xMin + fraction * (xMax - xMin);
    context.beginPath(); context.moveTo(x, margin.top); context.lineTo(x, margin.top + plotHeight); context.stroke();
    context.fillText(time.toFixed(1), x - 12, height - 8);
  }
  context.fillText(options.unit || '', 5, 12);

  if (options.horizontalLines) {
    for (const line of options.horizontalLines) {
      context.strokeStyle = line.color;
      context.setLineDash(line.dash || [6, 4]);
      context.beginPath(); context.moveTo(margin.left, toY(line.value)); context.lineTo(width - margin.right, toY(line.value)); context.stroke();
    }
  }
  for (const item of series) {
    context.strokeStyle = item.color;
    context.lineWidth = item.width || 2;
    context.setLineDash(item.dash || []);
    context.beginPath();
    responseHistory.forEach((point, index) => {
      const x = toX(point.time);
      const y = toY(item.value(point));
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }
  context.setLineDash([]);
}

function drawAllCharts() {
  drawChart($('angleChart'), [
    {value: point => point.target, color: '#34d399', dash: [7, 5], width: 2},
    {value: point => point.position, color: '#60a5fa', width: 3}
  ], {unit: 'deg', includeZero: true, minimumPadding: 2});
  drawChart($('velocityChart'), [
    {value: point => point.velocity, color: '#fbbf24', width: 2}
  ], {unit: 'deg/s', includeZero: true, minimumPadding: 2});
  drawChart($('torqueChart'), [
    {value: point => point.pdTorque, color: '#a78bfa', dash: [4, 4], width: 2},
    {value: point => point.gravityTorque, color: '#34d399', width: 2},
    {value: point => point.torque, color: '#f97316', width: 3}
  ], {
    unit: 'N·m', includeZero: true, extraValues: [-2, 2], minimumPadding: 0.2,
    horizontalLines: [{value: 2, color: '#ef4444'}, {value: -2, color: '#ef4444'}]
  });
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
async function toggleGravityCompensation() {
  try {
    const result = await post('/api/gravity-compensation', {
      enabled: !latest.gravity_compensation_enabled
    });
    $('pdStatus').textContent = result.gravity_compensation_enabled
      ? '已开启重力补偿：先托住重量，再由 PD 对准角度'
      : '已关闭重力补偿：现在由纯 PD 独自对抗重力';
    await refresh();
  } catch (error) { $('pdStatus').textContent = error.message; }
}
$('watchField').addEventListener('change', renderWatch);
$('applyTorque').addEventListener('click', () => setTorque($('torqueInput').value));
$('torqueInput').addEventListener('keydown', (event) => { if (event.key === 'Enter') setTorque(event.target.value); });
$('torqueSlider').addEventListener('change', (event) => setTorque(event.target.value));
document.querySelectorAll('[data-torque]').forEach((button) => button.addEventListener('click', () => setTorque(button.dataset.torque)));
$('applyPd').addEventListener('click', setPd);
$('toggleGravityCompensation').addEventListener('click', toggleGravityCompensation);
[$('targetInput'), $('kpInput'), $('kdInput')].forEach((input) => input.addEventListener('keydown', (event) => { if (event.key === 'Enter') setPd(); }));
$('toggleRecording').addEventListener('click', () => {
  recording = !recording;
  $('toggleRecording').textContent = recording ? '暂停记录' : '继续记录';
  $('curveStatus').textContent = recording ? '继续记录动态曲线' : '网页已暂停记录（MuJoCo 仿真仍可继续运行）';
});
$('clearCharts').addEventListener('click', clearResponseHistory);
$('reset').addEventListener('click', async () => {
  await post('/api/reset');
  clearResponseHistory();
  $('status').textContent = '已重置';
  $('pdStatus').textContent = '已重置姿态，PD 控制继续生效';
  await refresh();
});
window.addEventListener('resize', drawAllCharts);
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
        gravity_compensation: GravityCompensationSwitch | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.joint_id = joint_id
        self.actuator_id = actuator_id
        self.pd_controller = pd_controller
        self.gravity_compensation = gravity_compensation
        if self.pd_controller is not None and self.gravity_compensation is None:
            self.gravity_compensation = GravityCompensationSwitch(enabled=False)
        self._analysis_data = mujoco.MjData(model)
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
            with self._lock:
                current_bias_torque = float(self.data.qfrc_bias[qvel_address])
            compensation_torque = self.gravity_compensation.torque(
                current_bias_torque
            )
            output = self.pd_controller.compute(
                position_rad=float(snapshot["qpos"]),
                velocity_rad_s=float(snapshot["qvel"]),
                feedforward_torque_nm=compensation_torque,
            )
            settings = self.pd_controller.settings()
            target_rad = settings["target_position_rad"]
            with self._lock:
                mujoco.mj_resetData(self.model, self._analysis_data)
                self._analysis_data.qpos[qpos_address] = target_rad
                mujoco.mj_forward(self.model, self._analysis_data)
                target_hold_torque = float(
                    self._analysis_data.qfrc_bias[qvel_address]
                )
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
                    "pd_torque_nm": output.pd_torque_nm,
                    "gravity_compensation_torque_nm": (
                        output.feedforward_torque_nm
                    ),
                    "gravity_compensation_enabled": (
                        self.gravity_compensation.enabled
                    ),
                    "proportional_torque_nm": output.proportional_torque_nm,
                    "derivative_torque_nm": output.derivative_torque_nm,
                    "target_hold_torque_nm": target_hold_torque,
                    "bias_torque_nm": current_bias_torque,
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

    def set_gravity_compensation(self, enabled: bool) -> dict[str, bool]:
        if self.pd_controller is None or self.gravity_compensation is None:
            raise ValueError("当前不是 PD 控制模式")
        return {
            "gravity_compensation_enabled": (
                self.gravity_compensation.set_enabled(enabled)
            )
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
                    elif self.path == "/api/gravity-compensation":
                        self._send_json(
                            panel.set_gravity_compensation(body["enabled"])
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
