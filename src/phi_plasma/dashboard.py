"""Local dashboard for Phi-Plasma training runs and checkpoints.

Run with:
    PYTHONPATH=src python -m phi_plasma.dashboard --port 8765
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .constants import VOCAB_SIZE
from .train import build_model, pick_device


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phi Plasma Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f3;
      --panel: #ffffff;
      --ink: #172126;
      --muted: #64727a;
      --line: #d9e0dc;
      --teal: #137c72;
      --blue: #315f9f;
      --amber: #b56b17;
      --red: #b24034;
      --soft: #edf3f0;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1, h2, h3 { margin: 0; font-weight: 680; }
    h1 { font-size: 18px; }
    h2 { font-size: 15px; }
    h3 { font-size: 13px; color: var(--muted); }
    button, select, input, textarea {
      font: inherit;
      color: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      min-height: 34px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--teal);
      border-color: var(--teal);
      color: #fff;
    }
    button:disabled { opacity: .55; cursor: not-allowed; }
    select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 9px;
      outline: none;
    }
    textarea { resize: vertical; min-height: 96px; line-height: 1.35; }
    .shell {
      display: grid;
      grid-template-columns: minmax(210px, 280px) minmax(360px, 1fr) minmax(320px, 420px);
      gap: 12px;
      padding: 12px;
      min-height: calc(100vh - 58px);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
      overflow: hidden;
    }
    .panel-head {
      min-height: 48px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }
    .run-list { display: grid; gap: 8px; padding: 10px; }
    .run-item {
      width: 100%;
      text-align: left;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .run-item.active { border-color: var(--teal); background: var(--soft); }
    .run-name { font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .run-meta { color: var(--muted); font-size: 12px; }
    .dot { width: 8px; height: 8px; border-radius: 999px; align-self: center; background: var(--muted); }
    .dot.active { background: var(--teal); }
    .chart-wrap { padding: 12px; min-height: 360px; }
    canvas { width: 100%; height: 330px; display: block; }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 0 12px 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      min-width: 0;
    }
    .stat label { display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }
    .stat strong { font-size: 14px; overflow-wrap: anywhere; }
    .table-wrap { padding: 0 12px 12px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: right; padding: 7px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    .chat-body { display: grid; grid-template-rows: auto auto 1fr auto; min-height: calc(100vh - 108px); }
    .chat-controls { display: grid; gap: 9px; padding: 12px; border-bottom: 1px solid var(--line); }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .field { display: grid; gap: 4px; }
    .field label { font-size: 11px; color: var(--muted); }
    .messages { padding: 12px; overflow: auto; display: grid; align-content: start; gap: 10px; }
    .msg {
      border-radius: 8px;
      padding: 10px;
      line-height: 1.38;
      border: 1px solid var(--line);
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .msg.user { background: #eef4fb; border-color: #cbd9ea; }
    .msg.model { background: #f8f5ee; border-color: #eadcc6; }
    .msg small { display: block; color: var(--muted); margin-top: 6px; white-space: normal; }
    .prompt { display: grid; gap: 8px; padding: 12px; border-top: 1px solid var(--line); }
    .prompt-row { display: flex; justify-content: flex-end; gap: 8px; height: fit-content; }
    .health { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .empty { color: var(--muted); padding: 12px; font-size: 13px; }
    .ops, .telemetry { grid-column: 1 / -1; }
    .ops-body { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 1.25fr); gap: 0; }
    .op-column { padding: 12px; display: grid; align-content: start; gap: 12px; }
    .op-column + .op-column { border-left: 1px solid var(--line); }
    .op-column h4 { margin: 0; font-size: 13px; }
    .action-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .action-grid button { min-height: 38px; white-space: normal; line-height: 1.18; }
    .artifact-list { display: grid; gap: 8px; max-height: 260px; overflow: auto; }
    .artifact-item { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; padding: 9px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .artifact-title { font-weight: 650; font-size: 12px; overflow-wrap: anywhere; }
    .artifact-meta { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .telemetry-body { border-top: 1px solid var(--line); }
    .tabs { display: flex; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
    .tab-btn.active { border-color: var(--teal); background: var(--soft); }
    .tab-view { display: none; padding: 12px; }
    .tab-view.active { display: block; }
    .heatmap-tools { display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 10px; align-items: center; margin-bottom: 10px; }
    .heatmap-tools input[type=range] { width: 100%; }
    #gradHeatmap { height: 380px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .grad-detail { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .console { grid-column: 1 / -1; }
    .console-body { display: grid; grid-template-columns: minmax(320px, 440px) minmax(0, 1fr); min-height: 520px; }
    .command-form { padding: 12px; border-right: 1px solid var(--line); display: grid; align-content: start; gap: 10px; }
    .command-form textarea { min-height: 152px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .command-actions { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
    .job-list { display: grid; gap: 8px; max-height: 190px; overflow: auto; }
    .job-item { text-align: left; display: grid; gap: 4px; padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: #fff; }
    .job-item.active { border-color: var(--teal); background: var(--soft); }
    .job-title { font-weight: 650; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .job-meta { color: var(--muted); font-size: 11px; }
    .terminal { display: grid; grid-template-rows: auto 1fr; min-width: 0; }
    .terminal-head { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; gap: 8px; justify-content: space-between; align-items: center; }
    pre#commandLog { margin: 0; padding: 12px; overflow: auto; background: #101820; color: #d7ede8; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.42; white-space: pre-wrap; min-height: 440px; }
    @media (max-width: 1120px) {
      .shell { grid-template-columns: 240px minmax(0, 1fr); }
      .chat { grid-column: 1 / -1; }
      .chat-body { min-height: 560px; }
      .ops-body, .console-body { grid-template-columns: 1fr; }
      .op-column + .op-column, .command-form { border-left: 0; border-right: 0; border-bottom: 1px solid var(--line); }
      .action-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; height: auto; padding: 10px 12px; flex-direction: column; }
      .shell { grid-template-columns: 1fr; padding: 8px; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid-3, .action-grid, .heatmap-tools { grid-template-columns: 1fr; }
      canvas { height: 260px; }
      #gradHeatmap { height: 320px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Phi Plasma Dashboard</h1>
      <div class="health" id="health">Starting</div>
    </div>
    <button id="refreshBtn">Refresh</button>
  </header>
  <main class="shell">
    <aside class="panel">
      <div class="panel-head"><h2>Runs</h2><h3 id="runCount">0</h3></div>
      <div class="run-list" id="runList"></div>
    </aside>

    <section class="panel">
      <div class="panel-head"><h2 id="runTitle">Loss</h2><h3 id="updatedAt">-</h3></div>
      <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
      <div class="stats" id="stats"></div>
      <div class="table-wrap"><table id="metricTable"></table></div>
    </section>

    <section class="panel chat">
      <div class="panel-head"><h2>Checkpoint Chat</h2><h3 id="chatStatus">idle</h3></div>
      <div class="chat-body">
        <div class="chat-controls">
          <div class="field">
            <label for="checkpointSelect">Checkpoint</label>
            <select id="checkpointSelect"></select>
          </div>
          <div class="grid-3">
            <div class="field"><label for="maxNew">Tokens</label><input id="maxNew" type="number" min="1" max="512" value="64"></div>
            <div class="field"><label for="temperature">Temp</label><input id="temperature" type="number" min="0" max="3" step="0.05" value="0.8"></div>
            <div class="field"><label for="topK">Top-k</label><input id="topK" type="number" min="0" max="500" value="40"></div>
          </div>
        </div>
        <div class="messages" id="messages"></div>
        <form class="prompt" id="chatForm">
          <textarea id="promptInput" placeholder="Text prompt or token ids"></textarea>
          <div class="prompt-row">
            <button type="button" id="clearBtn">Clear</button>
            <button class="primary" type="submit" id="sendBtn">Send</button>
          </div>
        </form>
      </div>
    </section>
    <section class="panel ops">
      <div class="panel-head"><h2>Training & Data Control</h2><h3 id="dataStatus">data not scanned</h3></div>
      <div class="ops-body">
        <div class="op-column">
          <h4>Training Runs</h4>
          <div class="grid-3">
            <div class="field"><label for="runSteps">Steps Override</label><input id="runSteps" type="number" min="0" value="0"></div>
            <div class="field"><label for="evalPrompt">Eval Prompt</label><input id="evalPrompt" value="Solve step by step: If 3x + 7 = 31, what is x?"></div>
            <div class="field"><label for="sampleTokens">Sample Tokens</label><input id="sampleTokens" type="number" min="1" max="1024" value="256"></div>
          </div>
          <div class="action-grid">
            <button class="primary" type="button" data-preset="train_300m_distill">Start 300M Distill</button>
            <button type="button" data-preset="smoke_300m_distill">Smoke 1 Step</button>
            <button type="button" data-preset="train_byte_infer">Train Byte Model</button>
            <button type="button" data-preset="check_convergence_selected">Evaluate Selected Run</button>
            <button type="button" data-preset="sample_selected_checkpoint">Sample Selected Checkpoint</button>
            <button type="button" data-preset="test_suite">Run Test Suite</button>
          </div>
        </div>
        <div class="op-column">
          <h4>Training Data</h4>
          <div class="grid-3">
            <div class="field"><label for="teacherModel">Teacher</label><input id="teacherModel" value="qwen3.6:35b"></div>
            <div class="field"><label for="sampleCount">Synthetic Count</label><input id="sampleCount" type="number" min="1" value="50000"></div>
            <div class="field"><label for="syntheticWorkers">Workers</label><input id="syntheticWorkers" type="number" min="1" max="32" value="1"></div>
          </div>
          <div class="action-grid">
            <button class="primary" type="button" data-preset="generate_synthetic">Generate Synthetic</button>
            <button type="button" data-preset="generate_synthetic_smoke">Generate 20 Smoke</button>
            <button type="button" data-preset="pack_qwen_shards">Pack Qwen Shards</button>
            <button type="button" data-preset="install_scale_deps">Install Data Deps</button>
            <button type="button" data-preset="ollama_models">List Ollama Models</button>
            <button type="button" id="refreshDataBtn">Refresh Data</button>
          </div>
          <div class="artifact-list" id="artifactList"></div>
        </div>
      </div>
    </section>

    <section class="panel telemetry">
      <div class="panel-head"><h2>Training Telemetry</h2><h3 id="telemetryStatus">select a run</h3></div>
      <div class="telemetry-body">
        <div class="tabs">
          <button class="tab-btn active" type="button" data-tab="gradView">Gradient Heatmap</button>
          <button class="tab-btn" type="button" data-tab="lossTableView">Loss Table</button>
        </div>
        <div class="tab-view active" id="gradView">
          <div class="heatmap-tools">
            <input id="gradStepSlider" type="range" min="0" max="0" value="0">
            <span class="health" id="gradStepLabel">no gradient rows</span>
          </div>
          <canvas id="gradHeatmap"></canvas>
          <div class="grad-detail" id="gradDetail">No gradient telemetry yet. New training runs log per-module gradient norms.</div>
        </div>
        <div class="tab-view" id="lossTableView">
          <div class="table-wrap"><table id="telemetryTable"></table></div>
        </div>
      </div>
    </section>

    <section class="panel console">
      <div class="panel-head"><h2>Command Console</h2><h3 id="commandStatus">idle</h3></div>
      <div class="console-body">
        <div class="command-form">
          <div class="field">
            <label for="presetSelect">Preset</label>
            <select id="presetSelect"></select>
          </div>
          <div class="field">
            <label for="commandInput">Command</label>
            <textarea id="commandInput"></textarea>
          </div>
          <div class="command-actions">
            <button type="button" id="reloadPresetsBtn">Reload</button>
            <button type="button" id="stopJobBtn">Stop</button>
            <button class="primary" type="button" id="startJobBtn">Start</button>
          </div>
          <div class="field">
            <label>Jobs</label>
            <div class="job-list" id="jobList"></div>
          </div>
        </div>
        <div class="terminal">
          <div class="terminal-head"><strong id="activeJobTitle">No job selected</strong><span class="health" id="activeJobMeta">-</span></div>
          <pre id="commandLog"></pre>
        </div>
      </div>
    </section>
  </main>

<script>
const state = { runs: [], metrics: [], checkpoints: [], artifacts: [], selectedRun: null, presets: [], jobs: [], selectedJob: null, selectedGradIndex: 0 };
const $ = (id) => document.getElementById(id);

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  const n = Number(value);
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 100) return n.toFixed(1);
  return n.toFixed(digits);
}

function fmtTime(epoch) {
  if (!epoch) return '-';
  return new Date(epoch * 1000).toLocaleString();
}

function renderRuns() {
  $('runCount').textContent = String(state.runs.length);
  const list = $('runList');
  list.replaceChildren();
  if (!state.runs.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No metrics found';
    list.appendChild(empty);
    return;
  }
  for (const run of state.runs) {
    const btn = document.createElement('button');
    btn.className = 'run-item' + (run.name === state.selectedRun ? ' active' : '');
    btn.type = 'button';
    btn.addEventListener('click', () => selectRun(run.name));

    const name = document.createElement('div');
    name.className = 'run-name';
    name.textContent = run.name;
    const dot = document.createElement('div');
    dot.className = 'dot' + (run.status === 'active' ? ' active' : '');
    const meta = document.createElement('div');
    meta.className = 'run-meta';
    meta.textContent = `step ${run.latest_step || 0} | nll ${fmt(run.latest_nll)} | val ${fmt(run.latest_val_ppl, 2)}`;
    btn.append(name, dot, meta);
    list.appendChild(btn);
  }
}

function renderStats() {
  const run = state.runs.find(r => r.name === state.selectedRun);
  $('runTitle').textContent = run ? run.name : 'Loss';
  $('updatedAt').textContent = run ? fmtTime(run.updated_at) : '-';
  const stats = $('stats');
  stats.replaceChildren();
  const items = [
    ['step', run?.latest_step],
    ['nll', run?.latest_nll],
    ['loss', run?.latest_loss],
    ['val ppl', run?.latest_val_ppl],
    ['tok/s', run?.latest_tokens_s],
    ['world', run?.world_size],
    ['batch', run?.global_batch_size],
    ['ckpts', run?.checkpoint_count],
  ];
  for (const [label, value] of items) {
    const box = document.createElement('div');
    box.className = 'stat';
    const l = document.createElement('label');
    l.textContent = label;
    const s = document.createElement('strong');
    s.textContent = typeof value === 'number' ? fmt(value, label === 'val ppl' ? 2 : 3) : (value ?? '-');
    box.append(l, s);
    stats.appendChild(box);
  }
}

function drawChart() {
  const canvas = $('lossChart');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, rect.width, rect.height);

  const rows = state.metrics.filter(r => Number.isFinite(Number(r.step)) && (r.nll !== undefined || r.loss !== undefined));
  if (!rows.length) {
    ctx.fillStyle = '#64727a';
    ctx.font = '13px system-ui';
    ctx.fillText('No loss rows', 18, 28);
    return;
  }
  const pad = { l: 48, r: 18, t: 18, b: 32 };
  const w = rect.width - pad.l - pad.r;
  const h = rect.height - pad.t - pad.b;
  const steps = rows.map(r => Number(r.step));
  const values = rows.flatMap(r => [r.nll, r.loss].map(Number).filter(Number.isFinite));
  const minX = Math.min(...steps), maxX = Math.max(...steps);
  let minY = Math.min(...values), maxY = Math.max(...values);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const yPad = (maxY - minY) * 0.08;
  minY -= yPad; maxY += yPad;
  const x = (step) => pad.l + ((step - minX) / Math.max(1, maxX - minX)) * w;
  const y = (value) => pad.t + (1 - ((value - minY) / Math.max(1e-9, maxY - minY))) * h;

  ctx.strokeStyle = '#d9e0dc';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + (h * i / 4);
    ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + w, yy);
  }
  ctx.stroke();
  ctx.fillStyle = '#64727a';
  ctx.font = '11px system-ui';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const value = maxY - ((maxY - minY) * i / 4);
    ctx.fillText(fmt(value, 2), pad.l - 8, pad.t + (h * i / 4) + 4);
  }
  ctx.textAlign = 'center';
  ctx.fillText(String(minX), pad.l, rect.height - 10);
  ctx.fillText(String(maxX), pad.l + w, rect.height - 10);

  function line(key, color) {
    const pts = rows.filter(r => Number.isFinite(Number(r[key])));
    if (!pts.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((r, i) => {
      const xx = x(Number(r.step));
      const yy = y(Number(r[key]));
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  }
  line('loss', '#315f9f');
  line('nll', '#137c72');

  const vals = state.metrics.filter(r => r.val_ppl !== undefined && Number.isFinite(Number(r.step)));
  ctx.fillStyle = '#b56b17';
  for (const r of vals) {
    const trainAtStep = rows.find(rr => rr.step === r.step) || rows[rows.length - 1];
    const yy = y(Number(trainAtStep.nll ?? trainAtStep.loss));
    ctx.beginPath(); ctx.arc(x(Number(r.step)), yy, 4, 0, Math.PI * 2); ctx.fill();
  }
}

function fillMetricsTable(table, limit = 12) {
  table.replaceChildren();
  const cols = ['step', 'nll', 'loss', 'val_ppl', 'lr', 'tokens_s', 'grad_total_norm'];
  const head = document.createElement('tr');
  for (const c of cols) {
    const th = document.createElement('th'); th.textContent = c; head.appendChild(th);
  }
  table.appendChild(head);
  for (const row of state.metrics.slice(-limit).reverse()) {
    const tr = document.createElement('tr');
    for (const c of cols) {
      const td = document.createElement('td');
      td.textContent = c === 'step' ? (row[c] ?? '-') : fmt(row[c], c === 'lr' ? 6 : 3);
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
}

function renderTable() {
  fillMetricsTable($('metricTable'), 12);
  fillMetricsTable($('telemetryTable'), 80);
}

function gradientRows() {
  return state.metrics.filter(r => Number.isFinite(Number(r.step)) && Array.isArray(r.grad_groups) && r.grad_groups.length);
}

function renderGradDetail(rows) {
  if (!rows.length) {
    $('gradStepLabel').textContent = 'no gradient rows';
    $('gradDetail').textContent = 'No gradient telemetry yet. Start a new training run to log per-module gradient norms.';
    return;
  }
  state.selectedGradIndex = Math.max(0, Math.min(state.selectedGradIndex, rows.length - 1));
  const row = rows[state.selectedGradIndex];
  const top = [...row.grad_groups].sort((a, b) => Number(b.norm) - Number(a.norm)).slice(0, 8);
  $('gradStepLabel').textContent = `step ${row.step} | total ${fmt(row.grad_total_norm)} | max ${fmt(row.grad_max_abs)}`;
  $('gradDetail').textContent = top.map(g => `${g.group}: norm ${fmt(g.norm)} max ${fmt(g.max_abs)}`).join(' | ');
}

function drawGradHeatmap() {
  const canvas = $('gradHeatmap');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, rect.width, rect.height);

  const rows = gradientRows();
  const slider = $('gradStepSlider');
  slider.max = String(Math.max(0, rows.length - 1));
  slider.value = String(Math.max(0, Math.min(Number(slider.value || 0), rows.length - 1)));
  state.selectedGradIndex = Number(slider.value || 0);

  if (!rows.length) {
    ctx.fillStyle = '#64727a';
    ctx.font = '13px system-ui';
    ctx.fillText('No gradient heatmap rows yet', 18, 28);
    renderGradDetail(rows);
    return;
  }

  const groups = [];
  const seen = new Set();
  for (const row of rows) {
    for (const g of row.grad_groups) {
      if (!seen.has(g.group)) { seen.add(g.group); groups.push(g.group); }
    }
  }
  const values = [];
  const byStep = rows.map(row => {
    const m = new Map();
    for (const g of row.grad_groups) {
      const v = Number(g.log10_norm);
      if (Number.isFinite(v)) values.push(v);
      m.set(g.group, v);
    }
    return m;
  });
  let minV = Math.min(...values), maxV = Math.max(...values);
  if (!Number.isFinite(minV) || !Number.isFinite(maxV) || minV === maxV) { minV = -8; maxV = 0; }

  const pad = { l: 112, r: 18, t: 18, b: 34 };
  const w = Math.max(1, rect.width - pad.l - pad.r);
  const h = Math.max(1, rect.height - pad.t - pad.b);
  const cellW = Math.max(2, w / rows.length);
  const cellH = Math.max(4, h / groups.length);
  const color = (v) => {
    if (!Number.isFinite(v)) return '#eef2ef';
    const t = Math.max(0, Math.min(1, (v - minV) / Math.max(1e-9, maxV - minV)));
    const hue = 215 - 185 * t;
    const light = 88 - 38 * t;
    return `hsl(${hue} 70% ${light}%)`;
  };

  for (let xIdx = 0; xIdx < rows.length; xIdx++) {
    for (let yIdx = 0; yIdx < groups.length; yIdx++) {
      ctx.fillStyle = color(byStep[xIdx].get(groups[yIdx]));
      ctx.fillRect(pad.l + xIdx * cellW, pad.t + yIdx * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  const selectedX = pad.l + state.selectedGradIndex * cellW;
  ctx.strokeStyle = '#172126';
  ctx.lineWidth = 1;
  ctx.strokeRect(selectedX, pad.t, Math.max(2, cellW), h);

  ctx.fillStyle = '#64727a';
  ctx.font = '11px system-ui';
  ctx.textAlign = 'right';
  const labelEvery = Math.max(1, Math.ceil(groups.length / 18));
  groups.forEach((group, i) => {
    if (i % labelEvery === 0) ctx.fillText(group, pad.l - 8, pad.t + i * cellH + Math.min(cellH, 12));
  });
  ctx.textAlign = 'center';
  ctx.fillText(String(rows[0].step), pad.l, rect.height - 10);
  ctx.fillText(String(rows[rows.length - 1].step), pad.l + w, rect.height - 10);
  ctx.textAlign = 'left';
  ctx.fillText(`log10 grad norm ${fmt(minV, 2)} to ${fmt(maxV, 2)}`, pad.l, 12);
  $('telemetryStatus').textContent = `${rows.length} gradient rows`;
  renderGradDetail(rows);
}

function renderArtifacts() {
  const list = $('artifactList');
  list.replaceChildren();
  $('dataStatus').textContent = `${state.artifacts.length} artifacts`;
  if (!state.artifacts.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No training data artifacts found yet';
    list.appendChild(empty);
    return;
  }
  for (const item of state.artifacts) {
    const box = document.createElement('div');
    box.className = 'artifact-item';
    const title = document.createElement('div');
    title.className = 'artifact-title';
    title.textContent = `${item.kind} | ${item.path}`;
    const size = document.createElement('div');
    size.className = 'artifact-meta';
    size.textContent = `${fmt(item.size_mb, 2)} MB`;
    const meta = document.createElement('div');
    meta.className = 'artifact-meta';
    meta.textContent = item.summary || fmtTime(item.updated_at);
    box.append(title, size, meta);
    list.appendChild(box);
  }
}

async function refreshArtifacts() {
  const data = await getJSON('/api/data/artifacts');
  state.artifacts = data.artifacts || [];
  renderArtifacts();
}

function renderCheckpoints() {
  const select = $('checkpointSelect');
  const previous = select.value;
  select.replaceChildren();
  if (!state.checkpoints.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No checkpoints found';
    select.appendChild(opt);
    return;
  }
  for (const ckpt of state.checkpoints) {
    const opt = document.createElement('option');
    opt.value = ckpt.path;
    opt.textContent = `${ckpt.run} / ${ckpt.name}`;
    select.appendChild(opt);
  }
  if (previous && [...select.options].some(o => o.value === previous)) select.value = previous;
}

async function selectRun(name) {
  state.selectedRun = name;
  renderRuns();
  renderStats();
  const data = await getJSON(`/api/metrics?run=${encodeURIComponent(name)}&limit=2500`);
  state.metrics = data.rows || [];
  drawChart();
  renderTable();
  drawGradHeatmap();
}

async function refreshAll() {
  const [health, runs, checkpoints] = await Promise.all([
    getJSON('/api/health'), getJSON('/api/runs'), getJSON('/api/checkpoints')
  ]);
  $('health').textContent = `${health.device} | ${health.logs_dir}`;
  state.runs = runs.runs || [];
  state.checkpoints = checkpoints.checkpoints || [];
  if (!state.selectedRun && state.runs.length) state.selectedRun = state.runs[0].name;
  if (state.selectedRun && !state.runs.some(r => r.name === state.selectedRun)) state.selectedRun = state.runs[0]?.name || null;
  renderRuns();
  renderStats();
  renderCheckpoints();
  if (state.selectedRun) await selectRun(state.selectedRun);
}

function addMessage(kind, text, meta) {
  const box = document.createElement('div');
  box.className = `msg ${kind}`;
  box.textContent = text;
  if (meta) {
    const small = document.createElement('small');
    small.textContent = meta;
    box.appendChild(small);
  }
  $('messages').appendChild(box);
  $('messages').scrollTop = $('messages').scrollHeight;
  return box;
}

$('refreshBtn').addEventListener('click', () => refreshAll().catch(err => alert(err.message)));
$('clearBtn').addEventListener('click', () => $('messages').replaceChildren());
$('chatForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const checkpoint = $('checkpointSelect').value;
  const prompt = $('promptInput').value.trim();
  if (!checkpoint || !prompt) return;
  $('sendBtn').disabled = true;
  $('chatStatus').textContent = 'running';
  addMessage('user', prompt);
  const pending = addMessage('model', '...');
  try {
    const data = await getJSON('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        checkpoint,
        prompt,
        max_new_tokens: Number($('maxNew').value),
        temperature: Number($('temperature').value),
        top_k: Number($('topK').value),
      })
    });
    pending.textContent = data.text || data.generated_tokens.join(' ');
    const small = document.createElement('small');
    small.textContent = `${data.codec} | ${data.device} | tokens ${data.generated_tokens.join(' ')}`;
    pending.appendChild(small);
  } catch (err) {
    pending.textContent = err.message;
  } finally {
    $('sendBtn').disabled = false;
    $('chatStatus').textContent = 'idle';
  }
});


function shellQuote(value) {
  return "'" + String(value).replaceAll("'", "'\\''") + "'";
}

function presetVars() {
  const steps = Number($('runSteps').value || 0);
  const selectedCheckpoint = $('checkpointSelect').value || 'logs/a100_3gpu_plasma_300m_qwen_distill/ckpt_final.pt';
  const selectedRun = state.selectedRun || 'a100_3gpu_plasma_300m_qwen_distill';
  return {
    teacher: $('teacherModel').value || 'qwen3.6:35b',
    count: $('sampleCount').value || '50000',
    workers: $('syntheticWorkers').value || '1',
    steps: String(steps),
    maybe_steps: steps > 0 ? ` --steps ${steps}` : '',
    selected_run: selectedRun,
    selected_run_path: shellQuote(`logs/${selectedRun}`),
    selected_checkpoint: shellQuote(selectedCheckpoint),
    eval_prompt: shellQuote($('evalPrompt').value || 'Solve step by step: If 3x + 7 = 31, what is x?'),
    sample_tokens: $('sampleTokens').value || '256',
  };
}

function expandTemplate(text, vars) {
  return text.replace(/\{([a-z_]+)\}/g, (_, key) => Object.prototype.hasOwnProperty.call(vars, key) ? vars[key] : `{${key}}`);
}

function applyPreset() {
  const selected = state.presets.find(p => p.id === $('presetSelect').value);
  if (!selected) return;
  $('commandInput').value = expandTemplate(selected.command, presetVars());
}

function renderPresets() {
  const select = $('presetSelect');
  const old = select.value;
  select.replaceChildren();
  for (const preset of state.presets) {
    const opt = document.createElement('option');
    opt.value = preset.id;
    opt.textContent = preset.label;
    select.appendChild(opt);
  }
  if (old && [...select.options].some(o => o.value === old)) select.value = old;
  applyPreset();
}

function renderJobs() {
  const list = $('jobList');
  list.replaceChildren();
  if (!state.jobs.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No jobs yet';
    list.appendChild(empty);
    return;
  }
  for (const job of state.jobs) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'job-item' + (job.id === state.selectedJob ? ' active' : '');
    btn.addEventListener('click', () => { state.selectedJob = job.id; refreshJobLog(); renderJobs(); });
    const title = document.createElement('div');
    title.className = 'job-title';
    title.textContent = `${job.status} | ${job.label || job.id}`;
    const meta = document.createElement('div');
    meta.className = 'job-meta';
    meta.textContent = `${job.id} | code ${job.returncode ?? '-'} | ${fmtTime(job.started_at)}`;
    btn.append(title, meta);
    list.appendChild(btn);
  }
}

async function refreshPresets() {
  const data = await getJSON('/api/command/presets');
  state.presets = data.presets || [];
  renderPresets();
}

async function refreshJobs() {
  const data = await getJSON('/api/commands');
  state.jobs = data.jobs || [];
  if (!state.selectedJob && state.jobs.length) state.selectedJob = state.jobs[0].id;
  renderJobs();
  await refreshJobLog();
}

async function refreshJobLog() {
  if (!state.selectedJob) return;
  const data = await getJSON(`/api/command/log?id=${encodeURIComponent(state.selectedJob)}&limit=500`);
  $('commandLog').textContent = (data.lines || []).join('');
  const job = state.jobs.find(j => j.id === state.selectedJob) || data.job;
  if (job) {
    $('activeJobTitle').textContent = `${job.label || job.id}`;
    $('activeJobMeta').textContent = `${job.status} | code ${job.returncode ?? '-'} | ${job.command || ''}`;
    $('commandStatus').textContent = job.status;
  }
  $('commandLog').scrollTop = $('commandLog').scrollHeight;
}

async function startJob(commandOverride = null, labelOverride = null) {
  const command = (commandOverride || $('commandInput').value).trim();
  if (!command) return;
  $('startJobBtn').disabled = true;
  try {
    const data = await getJSON('/api/command/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, label: labelOverride || $('presetSelect').selectedOptions[0]?.textContent || 'custom' })
    });
    state.selectedJob = data.job.id;
    await refreshJobs();
    setTimeout(() => refreshArtifacts().catch(() => {}), 500);
  } catch (err) {
    alert(err.message);
  } finally {
    $('startJobBtn').disabled = false;
  }
}

async function startPreset(presetId) {
  const preset = state.presets.find(p => p.id === presetId);
  if (!preset) { alert(`Unknown preset: ${presetId}`); return; }
  $('presetSelect').value = preset.id;
  const command = expandTemplate(preset.command, presetVars());
  $('commandInput').value = command;
  await startJob(command, preset.label);
}

async function stopJob() {
  if (!state.selectedJob) return;
  await getJSON('/api/command/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: state.selectedJob })
  });
  await refreshJobs();
}

$('presetSelect').addEventListener('change', applyPreset);
for (const id of ['teacherModel', 'sampleCount', 'runSteps', 'syntheticWorkers', 'evalPrompt', 'sampleTokens']) {
  $(id).addEventListener('input', applyPreset);
}
$('reloadPresetsBtn').addEventListener('click', () => Promise.all([refreshPresets(), refreshJobs(), refreshArtifacts()]).catch(err => alert(err.message)));
$('refreshDataBtn').addEventListener('click', () => refreshArtifacts().catch(err => alert(err.message)));
$('startJobBtn').addEventListener('click', () => startJob());
$('stopJobBtn').addEventListener('click', stopJob);
$('gradStepSlider').addEventListener('input', () => { state.selectedGradIndex = Number($('gradStepSlider').value || 0); drawGradHeatmap(); });
document.querySelectorAll('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => startPreset(btn.dataset.preset));
});
document.querySelectorAll('[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-view').forEach(view => view.classList.toggle('active', view.id === btn.dataset.tab));
    drawGradHeatmap();
  });
});

window.addEventListener('resize', () => { drawChart(); drawGradHeatmap(); });
refreshAll().catch(err => { $('health').textContent = err.message; });
setInterval(() => refreshAll().catch(() => {}), 5000);
setInterval(() => refreshJobs().catch(() => {}), 2000);
setInterval(() => refreshArtifacts().catch(() => {}), 10000);
Promise.all([refreshPresets(), refreshJobs(), refreshArtifacts()]).catch(() => {});
</script>
</body>
</html>
"""


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return value


def select_device(spec: str) -> torch.device:
    spec = (spec or "cpu").lower()
    if spec == "auto":
        return pick_device()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    return torch.device(spec)


def read_jsonl(path: Path, limit: int = 2500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) > limit:
                rows = rows[-limit:]
    return rows


def latest_with(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if key in row:
            return row
    return None


def parse_step_from_name(path: Path) -> int | None:
    if path.name == "ckpt_final.pt":
        return None
    match = re.search(r"ckpt_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def safe_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def infer_arch(state_dict: dict[str, torch.Tensor]) -> str:
    keys = state_dict.keys()
    if any(k.startswith("backbone.") for k in keys):
        return "concentrate"
    if any("hecke" in k or "mass_log" in k or "V_net" in k for k in keys):
        return "plasma"
    return "vanilla"


@dataclass
class LoadedModel:
    path: Path
    mtime: float
    model: torch.nn.Module
    cfg: dict[str, Any]
    tokenizer: Any | None
    tokenizer_name: str | None


class CheckpointRuntime:
    def __init__(self, root: Path, device_spec: str):
        self.root = root.resolve()
        self.device = select_device(device_spec)
        self.lock = threading.Lock()
        self.loaded: LoadedModel | None = None

    def resolve_checkpoint(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("checkpoint path escapes dashboard root")
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {rel_path}")
        return path

    def load(self, rel_path: str) -> LoadedModel:
        path = self.resolve_checkpoint(rel_path)
        mtime = path.stat().st_mtime
        if self.loaded and self.loaded.path == path and self.loaded.mtime == mtime:
            return self.loaded

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        state_dict = strip_module_prefix(state_dict)
        cfg = dict(ckpt.get("cfg", {}))
        cfg.setdefault("arch", infer_arch(state_dict))
        cfg.setdefault("vocab_size", VOCAB_SIZE)

        model = build_model(cfg, torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        tokenizer, tokenizer_name = self.load_tokenizer(cfg)
        self.loaded = LoadedModel(path, mtime, model, cfg, tokenizer, tokenizer_name)
        return self.loaded

    @staticmethod
    def load_tokenizer(cfg: dict[str, Any]) -> tuple[Any | None, str | None]:
        target = cfg.get("tokenizer_path") or cfg.get("tokenizer_name")
        if not target:
            return None, None
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            return None, None
        try:
            return AutoTokenizer.from_pretrained(target), str(target)
        except Exception:
            return None, None

    def encode_prompt(self, loaded: LoadedModel, prompt: str) -> tuple[list[int], str]:
        if loaded.tokenizer is not None:
            ids = loaded.tokenizer.encode(prompt, add_special_tokens=False)
            return [int(x) for x in ids], f"tokenizer:{loaded.tokenizer_name}"

        parts = [p for p in re.split(r"[\s,]+", prompt.strip()) if p]
        if parts and all(p.isdigit() for p in parts):
            return [int(p) for p in parts], "token_ids"

        return list(prompt.encode("utf-8")) or [0], "byte_fallback"

    def decode_tokens(self, loaded: LoadedModel, tokens: list[int]) -> str:
        if loaded.tokenizer is not None:
            return loaded.tokenizer.decode(tokens, skip_special_tokens=False)
        if tokens and all(0 <= t < 256 for t in tokens):
            try:
                return bytes(tokens).decode("utf-8", errors="replace")
            except Exception:
                pass
        return " ".join(str(t) for t in tokens)

    def generate(self, rel_path: str, prompt: str, max_new_tokens: int,
                 temperature: float, top_k: int) -> dict[str, Any]:
        with self.lock:
            loaded = self.load(rel_path)
            model = loaded.model
            cfg = loaded.cfg
            input_tokens, codec = self.encode_prompt(loaded, prompt)
            vocab_size = int(cfg.get("vocab_size", VOCAB_SIZE))
            bad = [t for t in input_tokens if t < 0 or t >= vocab_size]
            if bad:
                raise ValueError(f"prompt token outside vocab range 0..{vocab_size - 1}: {bad[0]}")

            max_context = int(cfg.get("seq_len", 1024))
            max_new_tokens = max(1, min(int(max_new_tokens), 512))
            temperature = float(temperature)
            top_k = max(0, min(int(top_k), vocab_size))
            tokens = list(input_tokens)
            generated: list[int] = []

            for _ in range(max_new_tokens):
                context = tokens[-max_context:]
                idx = torch.tensor([context], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    out = model(idx)
                    logits = out["logits"] if isinstance(out, dict) else out
                    next_logits = logits[0, -1].float()
                    if temperature > 0:
                        next_logits = next_logits / max(temperature, 1e-6)
                        if top_k > 0 and top_k < next_logits.numel():
                            values, indices = torch.topk(next_logits, top_k)
                            masked = torch.full_like(next_logits, float("-inf"))
                            masked.scatter_(0, indices, values)
                            next_logits = masked
                        probs = torch.softmax(next_logits, dim=-1)
                        next_token = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        next_token = int(torch.argmax(next_logits).item())
                tokens.append(next_token)
                generated.append(next_token)

            return {
                "checkpoint": rel_path,
                "device": str(self.device),
                "codec": codec,
                "input_tokens": input_tokens,
                "generated_tokens": generated,
                "all_tokens": tokens,
                "text": self.decode_tokens(loaded, generated),
                "cfg": {
                    "arch": cfg.get("arch"),
                    "seq_len": cfg.get("seq_len"),
                    "vocab_size": cfg.get("vocab_size"),
                },
            }


COMMAND_PRESETS = [
    {
        "id": "train_byte_infer",
        "label": "Train byte inference model",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_byte_infer.yaml{maybe_steps}",
    },
    {
        "id": "train_300m_distill",
        "label": "Train 300M Qwen distill",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml{maybe_steps}",
    },
    {
        "id": "smoke_300m_distill",
        "label": "Smoke 300M Qwen distill",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml --steps 1",
    },
    {
        "id": "generate_synthetic",
        "label": "Generate Qwen reasoning JSONL",
        "command": "PYTHONPATH=src python scripts/generate_synthetic_reasoning.py --model {teacher} --seed-prompts prompts/reasoning_seeds.jsonl --out data/reasoning/qwen36_reasoning.jsonl --count {count} --workers {workers}",
    },
    {
        "id": "generate_synthetic_smoke",
        "label": "Generate 20 Qwen smoke records",
        "command": "PYTHONPATH=src python scripts/generate_synthetic_reasoning.py --model {teacher} --seed-prompts prompts/reasoning_seeds.jsonl --out data/reasoning/qwen36_reasoning_smoke.jsonl --count 20 --workers 1",
    },
    {
        "id": "pack_qwen_shards",
        "label": "Pack Qwen reasoning shards",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --input data/reasoning/qwen36_reasoning.jsonl --tokenizer Qwen/Qwen2.5-1.5B --out data/packed/qwen36_reasoning_qwen_tok --text-column text --validation-every 100",
    },
    {
        "id": "install_scale_deps",
        "label": "Install scale dependencies",
        "command": "python -m pip install -e '.[scale]'",
    },
    {
        "id": "check_convergence_300m",
        "label": "Check 300M convergence",
        "command": "PYTHONPATH=src python scripts/convergence_check.py --run logs/a100_3gpu_plasma_300m_qwen_distill",
    },
    {
        "id": "check_convergence_selected",
        "label": "Evaluate selected run convergence",
        "command": "PYTHONPATH=src python scripts/convergence_check.py --run {selected_run_path}",
    },
    {
        "id": "sample_300m",
        "label": "Sample 300M checkpoint",
        "command": "PYTHONPATH=src python scripts/generate.py --ckpt logs/a100_3gpu_plasma_300m_qwen_distill/ckpt_final.pt --prompt 'Solve step by step: If 3x + 7 = 31, what is x?' --device cuda --max-new-tokens 256",
    },
    {
        "id": "sample_selected_checkpoint",
        "label": "Deploy/sample selected checkpoint",
        "command": "PYTHONPATH=src python scripts/generate.py --ckpt {selected_checkpoint} --prompt {eval_prompt} --device cuda --max-new-tokens {sample_tokens}",
    },
    {
        "id": "ollama_models",
        "label": "List Ollama models",
        "command": "ollama list",
    },
    {
        "id": "test_suite",
        "label": "Run tests",
        "command": "PYTHONPATH=src pytest tests/",
    },
]


@dataclass
class CommandJob:
    id: str
    label: str
    command: str
    started_at: float
    status: str = "running"
    returncode: int | None = None
    ended_at: float | None = None
    pid: int | None = None
    log_path: Path | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    process: subprocess.Popen | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "command": self.command,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "returncode": self.returncode,
            "pid": self.pid,
            "log_path": str(self.log_path) if self.log_path else None,
        }


class CommandManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.log_dir = self.root / "logs" / "dashboard_commands"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, CommandJob] = {}

    def presets(self) -> list[dict[str, str]]:
        return list(COMMAND_PRESETS)

    def start(self, command: str, label: str = "custom") -> CommandJob:
        command = command.strip()
        if not command:
            raise ValueError("command is empty")
        job_id = uuid.uuid4().hex[:10]
        log_path = self.log_dir / f"{job_id}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.root / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(self.root / "src")
        job = CommandJob(
            id=job_id,
            label=label or "custom",
            command=command,
            started_at=time.time(),
            log_path=log_path,
        )
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"$ {command}\n")
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(self.root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        job.process = process
        job.pid = process.pid
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._reader, args=(job,), daemon=True).start()
        return job

    def _reader(self, job: CommandJob) -> None:
        assert job.process is not None
        assert job.log_path is not None
        try:
            with job.log_path.open("a", encoding="utf-8", errors="replace") as f:
                if job.process.stdout is not None:
                    for line in job.process.stdout:
                        with self.lock:
                            job.lines.append(line)
                        f.write(line)
                        f.flush()
                rc = job.process.wait()
                with self.lock:
                    prior_status = job.status
                    job.returncode = rc
                    job.ended_at = time.time()
                    if rc == 0:
                        job.status = "completed"
                    elif prior_status == "stopping":
                        job.status = "stopped"
                    else:
                        job.status = "failed"
                f.write(f"\n[exit {rc}]\n")
        except Exception as exc:
            with self.lock:
                job.returncode = -1
                job.ended_at = time.time()
                job.status = "failed"
                job.lines.append(f"[dashboard error] {exc}\n")

    def stop(self, job_id: str) -> CommandJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        if job.process and job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            with self.lock:
                job.status = "stopping"
                job.lines.append("[dashboard] sent SIGTERM\n")
        return job

    def summaries(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return [j.summary() for j in jobs]

    def log(self, job_id: str, limit: int = 500) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"unknown job: {job_id}")
            lines = list(job.lines)[-max(1, min(limit, 2000)):]
            summary = job.summary()
        if not lines and job.log_path and job.log_path.exists():
            lines = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-limit:]
        return {"job": summary, "lines": lines}


class DashboardState:
    def __init__(self, root: Path, logs_dir: Path, device_spec: str):
        self.root = root.resolve()
        self.logs_dir = logs_dir.resolve()
        self.runtime = CheckpointRuntime(self.root, device_spec)
        self.commands = CommandManager(self.root)

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def run_dirs(self) -> list[Path]:
        if not self.logs_dir.exists():
            return []
        dirs = [p for p in self.logs_dir.iterdir() if p.is_dir() and (p / "metrics.jsonl").exists()]
        return sorted(dirs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)

    def runs(self) -> list[dict[str, Any]]:
        result = []
        now = time.time()
        for run_dir in self.run_dirs():
            metrics_path = run_dir / "metrics.jsonl"
            rows = read_jsonl(metrics_path)
            latest = rows[-1] if rows else {}
            train = latest_with(rows, "nll") or latest_with(rows, "loss") or {}
            val = latest_with(rows, "val_ppl") or {}
            checkpoints = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            mtime = metrics_path.stat().st_mtime
            result.append({
                "name": run_dir.name,
                "path": self.rel(run_dir),
                "metrics_count": len(rows),
                "latest_step": latest.get("step") or train.get("step") or val.get("step"),
                "latest_loss": train.get("loss"),
                "latest_nll": train.get("nll"),
                "latest_val_ppl": val.get("val_ppl"),
                "latest_tokens_s": train.get("tokens_s"),
                "world_size": train.get("world_size"),
                "global_batch_size": train.get("global_batch_size"),
                "updated_at": mtime,
                "status": "active" if now - mtime < 180 else "idle",
                "checkpoint_count": len(checkpoints),
            })
        return result

    def metrics(self, run_name: str, limit: int) -> list[dict[str, Any]]:
        run_dir = (self.logs_dir / run_name).resolve()
        if run_dir != self.logs_dir and self.logs_dir not in run_dir.parents:
            raise ValueError("run path escapes logs dir")
        return read_jsonl(run_dir / "metrics.jsonl", limit=limit)

    def checkpoints(self) -> list[dict[str, Any]]:
        if not self.logs_dir.exists():
            return []
        rows = []
        for path in sorted(self.logs_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            rows.append({
                "name": path.name,
                "path": self.rel(path),
                "run": path.parent.name,
                "step": parse_step_from_name(path),
                "size_mb": path.stat().st_size / (1024 * 1024),
                "updated_at": path.stat().st_mtime,
            })
        return rows

    def data_artifacts(self) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []

        def add_file(kind: str, path: Path, summary: str | None = None, records: int | None = None) -> None:
            try:
                stat = path.stat()
            except OSError:
                return
            artifacts.append({
                "kind": kind,
                "path": self.rel(path),
                "size_mb": stat.st_size / (1024 * 1024),
                "updated_at": stat.st_mtime,
                "records": records,
                "summary": summary or f"updated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}",
            })

        reasoning_dir = self.root / "data" / "reasoning"
        if reasoning_dir.exists():
            for path in sorted(reasoning_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
                records = count_lines(path)
                add_file("synthetic-jsonl", path, summary=f"{records} records | JSONL teacher traces", records=records)

        packed_dir = self.root / "data" / "packed"
        if packed_dir.exists():
            for meta_path in sorted(packed_dir.rglob("meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                meta = safe_json(meta_path)
                dataset_dir = meta_path.parent
                splits = meta.get("splits", {}) if isinstance(meta, dict) else {}
                train = splits.get("train", {}) if isinstance(splits, dict) else {}
                val = splits.get("validation", {}) if isinstance(splits, dict) else {}
                tokens = int(train.get("tokens", 0) or 0) + int(val.get("tokens", 0) or 0)
                docs = int(train.get("docs", 0) or 0) + int(val.get("docs", 0) or 0)
                artifacts.append({
                    "kind": "packed-shards",
                    "path": self.rel(dataset_dir),
                    "size_mb": directory_size(dataset_dir) / (1024 * 1024),
                    "updated_at": meta_path.stat().st_mtime,
                    "records": docs,
                    "summary": f"{docs} docs | {tokens} tokens | tokenizer {meta.get('tokenizer_name') or 'byte'}",
                })

        seed_path = self.root / "prompts" / "reasoning_seeds.jsonl"
        if seed_path.exists():
            records = count_lines(seed_path)
            add_file("seed-prompts", seed_path, summary=f"{records} prompt seeds", records=records)

        cache_dir = self.root / ".token_cache"
        if cache_dir.exists():
            for path in sorted(cache_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
                add_file("token-cache", path, summary="local token cache")

        artifacts.sort(key=lambda item: float(item.get("updated_at", 0)), reverse=True)
        return artifacts[:200]

    def health(self) -> dict[str, Any]:
        cuda_devices = []
        if torch.cuda.is_available():
            cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return {
            "device": str(self.runtime.device),
            "logs_dir": self.rel(self.logs_dir),
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": cuda_devices,
        }


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body too large")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(body)

    def require_local_command_client(self) -> None:
        host = self.client_address[0]
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            if host == "localhost":
                return
        raise PermissionError("command console is restricted to localhost clients")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/health":
                self.send_json(self.state.health())
                return
            if parsed.path == "/api/runs":
                self.send_json({"runs": self.state.runs()})
                return
            if parsed.path == "/api/metrics":
                query = parse_qs(parsed.query)
                run = query.get("run", [""])[0]
                limit = int(query.get("limit", ["2500"])[0])
                self.send_json({"rows": self.state.metrics(run, max(1, min(limit, 20000)))})
                return
            if parsed.path == "/api/checkpoints":
                self.send_json({"checkpoints": self.state.checkpoints()})
                return
            if parsed.path == "/api/data/artifacts":
                self.require_local_command_client()
                self.send_json({"artifacts": self.state.data_artifacts()})
                return
            if parsed.path == "/api/command/presets":
                self.require_local_command_client()
                self.send_json({"presets": self.state.commands.presets()})
                return
            if parsed.path == "/api/commands":
                self.require_local_command_client()
                self.send_json({"jobs": self.state.commands.summaries()})
                return
            if parsed.path == "/api/command/log":
                self.require_local_command_client()
                query = parse_qs(parsed.query)
                job_id = query.get("id", [""])[0]
                limit = int(query.get("limit", ["500"])[0])
                self.send_json(self.state.commands.log(job_id, limit=limit))
                return
            self.send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/chat":
                body = self.read_json()
                result = self.state.runtime.generate(
                    rel_path=str(body.get("checkpoint", "")),
                    prompt=str(body.get("prompt", "")),
                    max_new_tokens=int(body.get("max_new_tokens", 64)),
                    temperature=float(body.get("temperature", 0.8)),
                    top_k=int(body.get("top_k", 40)),
                )
                self.send_json(result)
                return
            if parsed.path == "/api/command/start":
                self.require_local_command_client()
                body = self.read_json()
                job = self.state.commands.start(
                    command=str(body.get("command", "")),
                    label=str(body.get("label", "custom")),
                )
                self.send_json({"job": job.summary()})
                return
            if parsed.path == "/api/command/stop":
                self.require_local_command_client()
                body = self.read_json()
                job = self.state.commands.stop(str(body.get("id", "")))
                self.send_json({"job": job.summary()})
                return
            self.send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local Phi-Plasma training dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=".")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--device", default="cpu",
                    help="checkpoint chat device: cpu, auto, cuda, mps, or explicit torch device")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    logs_dir = (root / args.logs).resolve()
    DashboardHandler.state = DashboardState(root, logs_dir, args.device)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[dashboard] {url}  logs={logs_dir}  device={DashboardHandler.state.runtime.device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
