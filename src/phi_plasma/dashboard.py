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
import sys
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
      --panel-elev: #ffffff;
      --ink: #172126;
      --ink-strong: #0d1518;
      --muted: #64727a;
      --line: #d9e0dc;
      --line-strong: #b9c4be;
      --teal: #137c72;
      --teal-soft: #1a8f83;
      --blue: #315f9f;
      --amber: #b56b17;
      --red: #b24034;
      --green: #2a7d3e;
      --soft: #edf3f0;
      --header-bg: rgba(255,255,255,0.92);
      --input-bg: #ffffff;
      --code-bg: #101820;
      --code-fg: #d7ede8;
      --chip-bg: #eef3f0;
      --user-msg-bg: #eef4fb;
      --user-msg-bd: #cbd9ea;
      --model-msg-bg: #f8f5ee;
      --model-msg-bd: #eadcc6;
    }
    [data-theme="dark"] {
      color-scheme: dark;
      --bg: #0f1518;
      --panel: #16201f;
      --panel-elev: #1c2826;
      --ink: #dfe8e3;
      --ink-strong: #f4f7f5;
      --muted: #8b9a93;
      --line: #2a3733;
      --line-strong: #3a4a44;
      --teal: #4fbfae;
      --teal-soft: #5fd2c0;
      --blue: #7ba6e8;
      --amber: #e9b97e;
      --red: #e9817b;
      --green: #74d189;
      --soft: #1f2c29;
      --header-bg: rgba(15,21,24,0.92);
      --input-bg: #1c2826;
      --code-bg: #0a1112;
      --code-fg: #cfe5dd;
      --chip-bg: #1c2826;
      --user-msg-bg: #1c2a36;
      --user-msg-bd: #2d4860;
      --model-msg-bg: #2b2620;
      --model-msg-bd: #4a3f2c;
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
      background: var(--header-bg);
      backdrop-filter: blur(8px);
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
      background: var(--input-bg);
      padding: 8px 9px;
      outline: none;
      color: var(--ink);
    }
    select:focus, input:focus, textarea:focus { border-color: var(--teal); }
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
      background: var(--panel);
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
    .artifact-item { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; padding: 9px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .artifact-title { font-weight: 650; font-size: 12px; overflow-wrap: anywhere; }
    .artifact-meta { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .telemetry-body { border-top: 1px solid var(--line); }
    .tabs { display: flex; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
    .tab-btn.active { border-color: var(--teal); background: var(--soft); }
    .tab-view { display: none; padding: 12px; }
    .tab-view.active { display: block; }
    .heatmap-tools { display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 10px; align-items: center; margin-bottom: 10px; }
    .heatmap-tools input[type=range] { width: 100%; }
    #gradHeatmap { height: 380px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .grad-detail { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .console { grid-column: 1 / -1; }
    .console-body { display: grid; grid-template-columns: minmax(320px, 440px) minmax(0, 1fr); min-height: 520px; }
    .command-form { padding: 12px; border-right: 1px solid var(--line); display: grid; align-content: start; gap: 10px; }
    .command-form textarea { min-height: 152px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .command-actions { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
    .command-form > .field { min-height: 0; }
    .job-list { display: flex; flex-direction: column; gap: 8px; max-height: 240px; min-height: 80px; overflow-y: auto; overflow-x: hidden; }
    .job-item { text-align: left; display: grid; gap: 4px; padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); }
    .job-item.active { border-color: var(--teal); background: var(--soft); }
    .job-title { font-weight: 650; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .job-meta { color: var(--muted); font-size: 11px; }
    .terminal { display: grid; grid-template-rows: auto 1fr; min-width: 0; }
    .terminal-head { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; gap: 8px; justify-content: space-between; align-items: center; }
    pre#commandLog { margin: 0; padding: 12px; overflow: auto; background: var(--code-bg); color: var(--code-fg); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.42; white-space: pre-wrap; min-height: 440px; }
    .header-actions { display: flex; gap: 8px; align-items: center; }
    .chip { background: var(--chip-bg); border: 1px solid var(--line); color: var(--ink); border-radius: 999px; padding: 4px 10px; font-size: 12px; }
    #themeToggleBtn { min-width: 38px; padding: 0 10px; }
    .sidebar { display: flex; flex-direction: column; min-height: 0; }
    .sidebar-section { padding: 10px 12px 6px; display: grid; gap: 8px; }
    .inline-row { display: flex; gap: 6px; }
    .inline-row input { flex: 1 1 auto; min-width: 0; }
    .inline-row button { white-space: nowrap; }
    .panel-divider { height: 1px; background: var(--line); margin: 4px 0; }
    .pipeline { padding: 6px 10px 10px; display: grid; gap: 6px; }
    .pipeline-item { display: grid; grid-template-columns: 14px 1fr auto; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); align-items: center; }
    .pipeline-item .dot { width: 10px; height: 10px; border-radius: 999px; background: var(--muted); }
    .pipeline-item[data-status="pending"] .dot { background: var(--muted); }
    .pipeline-item[data-status="in_progress"] .dot { background: var(--amber); box-shadow: 0 0 0 4px rgba(181,107,23,.18); }
    .pipeline-item[data-status="done"] .dot { background: var(--green); }
    .pipeline-item[data-status="done"] { border-color: color-mix(in srgb, var(--green) 30%, var(--line)); }
    .pipeline-item[data-status="in_progress"] { border-color: color-mix(in srgb, var(--amber) 35%, var(--line)); }
    .pipeline-label { font-size: 12px; font-weight: 650; color: var(--ink); }
    .pipeline-detail { font-size: 11px; color: var(--muted); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pipeline-step-num { font-size: 10px; color: var(--muted); }
    button[data-gated="true"] { opacity: .42; cursor: not-allowed; }
    button[data-gated="true"]::after { content: " ⛔"; }
    .stage-banner { padding: 8px 12px; background: var(--soft); border-bottom: 1px solid var(--line); font-size: 12px; color: var(--ink); display: flex; gap: 12px; flex-wrap: wrap; }
    .stage-banner strong { color: var(--ink-strong); }
    .stage-banner .sep { color: var(--muted); }

    /* ===== Tabbed pipeline layout ===== */
    .tabnav {
      position: sticky; top: 58px; z-index: 1;
      background: var(--header-bg);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
      display: flex; align-items: center; gap: 4px;
      padding: 8px 16px;
      overflow-x: auto;
      scrollbar-width: none;
    }
    .tabnav::-webkit-scrollbar { display: none; }
    .tab-pill {
      display: inline-flex; align-items: center; gap: 8px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 14px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0;
      white-space: nowrap;
      transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
    }
    .tab-pill:hover { color: var(--ink); border-color: var(--line-strong); }
    .tab-pill[aria-current="true"] {
      background: var(--soft);
      color: var(--ink-strong);
      border-color: var(--teal);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--teal) 18%, transparent);
    }
    .tab-pill .tab-step {
      display: inline-flex; align-items: center; justify-content: center;
      width: 20px; height: 20px; border-radius: 50%;
      background: var(--chip-bg); color: var(--ink);
      font-size: 11px; font-weight: 700;
    }
    .tab-pill[aria-current="true"] .tab-step { background: var(--teal); color: #fff; }
    .tab-pill .tab-label { font-weight: 600; }
    .tab-pill .tab-dot {
      width: 8px; height: 8px; border-radius: 999px; background: var(--muted);
      transition: background 120ms ease;
    }
    .tab-pill .tab-dot[data-status="pending"] { background: var(--muted); opacity: .55; }
    .tab-pill .tab-dot[data-status="in_progress"] { background: var(--amber); box-shadow: 0 0 0 3px color-mix(in srgb, var(--amber) 25%, transparent); }
    .tab-pill .tab-dot[data-status="done"] { background: var(--green); }
    .tab-pill.ghost { color: var(--muted); border-style: dashed; }
    .tab-pill.ghost[aria-current="true"] { color: var(--ink-strong); border-style: solid; }
    .tab-sep { color: var(--muted); opacity: .6; font-size: 14px; }
    .tabnav-spacer { flex: 1 1 auto; }

    .warning-bar {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 16px; font-size: 13px;
      background: color-mix(in srgb, var(--amber) 14%, var(--panel));
      border-bottom: 1px solid color-mix(in srgb, var(--amber) 30%, var(--line));
      color: var(--ink);
    }
    .warning-bar[hidden] { display: none; }
    .warning-bar-title { color: var(--amber); font-weight: 700; }
    .warning-bar #warningBarText { flex: 1 1 auto; }
    .warning-bar button {
      background: transparent; border: 0; color: var(--muted); cursor: pointer;
      font-size: 18px; line-height: 1; min-height: 0; padding: 4px 8px;
    }

    .workspace {
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 16px 40px;
      display: block;
    }
    .tab-page { display: none; }
    .tab-page[data-active="true"] { display: grid; gap: 16px; }
    .page-head { display: grid; gap: 4px; padding: 4px 2px 8px; }
    .page-head h2 { font-size: 18px; color: var(--ink-strong); }
    .page-sub { font-size: 13px; color: var(--muted); margin: 0; line-height: 1.5; }
    .page-sub code { background: var(--chip-bg); border-radius: 4px; padding: 1px 5px; font-size: 12px; }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .card h3 { font-size: 14px; color: var(--ink-strong); margin: 0; }
    .card .card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .cards-2col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    @media (max-width: 880px) { .cards-2col { grid-template-columns: 1fr; } }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .action-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .action-row button { min-height: 36px; }
    .muted { color: var(--muted); }
    .small { font-size: 11px; }
    .hint { color: var(--muted); font-size: 12px; }

    .train-grid { display: grid; grid-template-columns: minmax(220px, 320px) minmax(0, 1fr); gap: 14px; }
    @media (max-width: 880px) { .train-grid { grid-template-columns: 1fr; } }
    .train-runs .run-list { max-height: 320px; overflow: auto; }

    .subtabs { display: flex; gap: 8px; flex-wrap: wrap; }
    .subtab-btn { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; min-height: 32px; cursor: pointer; }
    .subtab-btn.active { border-color: var(--teal); background: var(--soft); color: var(--ink-strong); }
    .subtab-view { display: none; }
    .subtab-view.active { display: block; }

    .chat-card .chat-body { display: grid; grid-template-rows: auto 1fr auto; gap: 8px; min-height: 360px; }

    .console-body { display: grid; grid-template-columns: minmax(280px, 380px) minmax(0, 1fr); gap: 14px; align-items: stretch; }
    @media (max-width: 880px) { .console-body { grid-template-columns: 1fr; } }
    .console-body .command-form { padding: 12px; }
    .console-body .terminal { padding: 0; display: grid; grid-template-rows: auto 1fr; min-height: 440px; }
    .console-body .terminal-head { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; gap: 8px; justify-content: space-between; align-items: center; }
    .console-body pre#commandLog { min-height: 380px; max-height: 70vh; }

    .stage-progress { display: grid; gap: 6px; }
    .stage-progress-row { display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline; font-size: 13px; color: var(--ink); }
    .stage-progress-row .muted { font-size: 11px; }
    .stage-progress-row strong { font-variant-numeric: tabular-nums; }
    .progress-bar { height: 6px; background: var(--chip-bg); border-radius: 999px; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, var(--teal), var(--teal-soft)); width: 0; transition: width 400ms ease; }

    button[data-stage-warn="true"] {
      box-shadow: 0 0 0 2px color-mix(in srgb, var(--amber) 30%, transparent);
    }
    button[data-stage-warn="true"]::before { content: "⚠ "; }
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
    <div class="header-actions">
      <span class="chip" id="sessionChip">session: -</span>
      <button id="themeToggleBtn" title="Toggle dark/light mode">☾</button>
      <button id="refreshBtn">Refresh</button>
    </div>
  </header>
  <nav class="tabnav" id="tabnav" role="tablist" aria-label="Pipeline stages">
    <button class="tab-pill" type="button" role="tab" data-tab="setup" data-stage-key="setup">
      <span class="tab-step">1</span>
      <span class="tab-label">Setup</span>
      <span class="tab-dot" data-status="pending"></span>
    </button>
    <span class="tab-sep">›</span>
    <button class="tab-pill" type="button" role="tab" data-tab="generate" data-stage-key="generate">
      <span class="tab-step">2</span>
      <span class="tab-label">Generate</span>
      <span class="tab-dot" data-status="pending"></span>
    </button>
    <span class="tab-sep">›</span>
    <button class="tab-pill" type="button" role="tab" data-tab="tokenize" data-stage-key="tokenize">
      <span class="tab-step">3</span>
      <span class="tab-label">Tokenize</span>
      <span class="tab-dot" data-status="pending"></span>
    </button>
    <span class="tab-sep">›</span>
    <button class="tab-pill" type="button" role="tab" data-tab="train" data-stage-key="train">
      <span class="tab-step">4</span>
      <span class="tab-label">Train</span>
      <span class="tab-dot" data-status="pending"></span>
    </button>
    <span class="tab-sep">›</span>
    <button class="tab-pill" type="button" role="tab" data-tab="evaluate" data-stage-key="evaluate">
      <span class="tab-step">5</span>
      <span class="tab-label">Evaluate</span>
      <span class="tab-dot" data-status="pending"></span>
    </button>
    <span class="tabnav-spacer"></span>
    <button class="tab-pill ghost" type="button" role="tab" data-tab="console">
      <span class="tab-label">Console</span>
    </button>
  </nav>

  <div class="warning-bar" id="warningBar" hidden>
    <strong class="warning-bar-title">Heads up:</strong>
    <span id="warningBarText">…</span>
    <button type="button" id="warningBarClose" aria-label="Dismiss">×</button>
  </div>

  <main class="workspace">

    <!-- ===== Tab 1: Setup ===== -->
    <section class="tab-page" data-tab="setup">
      <div class="page-head">
        <h2>Setup — session, teacher, dependencies</h2>
        <p class="page-sub">Pick or create a session, point at your Ollama teacher, install scale deps.
        Each session keeps its own data, packed shards, checkpoints, and eval results under <code>sessions/&lt;name&gt;</code>.</p>
      </div>
      <div class="cards-2col">
        <div class="card">
          <h3>Session</h3>
          <div class="field">
            <label for="sessionSelect">Active session</label>
            <select id="sessionSelect"></select>
          </div>
          <div class="field">
            <label for="newSessionName">Create new session</label>
            <div class="inline-row">
              <input id="newSessionName" placeholder="my-experiment-01">
              <button type="button" id="createSessionBtn">+ New</button>
            </div>
          </div>
          <div class="hint" id="activeSessionLabel">default</div>
        </div>
        <div class="card">
          <h3>Teacher (Ollama)</h3>
          <div class="field">
            <label for="teacherModel">Local model</label>
            <select id="teacherModel"></select>
          </div>
          <div class="grid-2">
            <div class="field"><label for="sampleCount">Synthetic Count</label><input id="sampleCount" type="number" min="1" value="50000"></div>
            <div class="field"><label for="syntheticWorkers">Workers</label><input id="syntheticWorkers" type="number" min="1" max="32" value="1"></div>
          </div>
          <div class="action-row">
            <button type="button" data-preset="ollama_models">List Models</button>
            <button type="button" data-preset="install_scale_deps">Install Scale Deps</button>
            <button type="button" data-preset="test_suite">Run Tests</button>
          </div>
        </div>
      </div>
    </section>

    <!-- ===== Tab 2: Generate ===== -->
    <section class="tab-page" data-tab="generate">
      <div class="page-head">
        <h2>Generate — synthetic teacher traces</h2>
        <p class="page-sub">Calls the local Ollama teacher for each seed prompt and writes JSONL into the active session.
        Output goes to <code id="genOutPath">sessions/&lt;active&gt;/data/reasoning.jsonl</code>.</p>
      </div>
      <div class="card">
        <div class="action-row">
          <button class="primary" type="button" data-preset="generate_synthetic" data-stage="generate">Generate Synthetic ({count})</button>
          <button type="button" data-preset="generate_synthetic_smoke" data-stage="generate">Smoke 20</button>
          <button type="button" id="refreshDataBtn">Refresh</button>
        </div>
        <div class="stage-progress" id="generateProgress">
          <div class="stage-progress-row">
            <span class="muted">records:</span><strong id="genRecords">0</strong>
            <span class="muted">rate:</span><strong id="genRate">-</strong>
            <span class="muted">eta:</span><strong id="genEta">-</strong>
            <span class="muted">avg chars:</span><strong id="genAvgChars">-</strong>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="genProgressFill" style="width:0%"></div></div>
        </div>
        <div class="hint">Watch the Console tab for the live log. Heartbeats also land in <code>&lt;out&gt;.progress.json</code>.</div>
      </div>
      <div class="card">
        <h3>Recent JSONL outputs</h3>
        <div class="artifact-list" id="artifactListGenerate"></div>
      </div>
    </section>

    <!-- ===== Tab 3: Tokenize ===== -->
    <section class="tab-page" data-tab="tokenize">
      <div class="page-head">
        <h2>Tokenize — pack JSONL into training shards</h2>
        <p class="page-sub">Streams the session's JSONL (or a Hugging Face dataset) through a tokenizer and writes
        packed <code>train.bin</code> / <code>validation.bin</code> + <code>meta.json</code> into
        <code id="tokOutPath">sessions/&lt;active&gt;/packed/</code>.</p>
      </div>
      <div class="card">
        <h3>Pack the session's synthetic data</h3>
        <div class="action-row">
          <button class="primary" type="button" data-preset="pack_session_qwen" data-stage="tokenize">Pack (Qwen tokenizer)</button>
          <button type="button" data-preset="pack_session_byte" data-stage="tokenize">Pack (byte tokenizer)</button>
        </div>
      </div>
      <div class="card">
        <h3>Or pack a public dataset</h3>
        <p class="muted small">Caps at a sensible <code>--max-docs</code> so a smoke pack finishes in minutes.</p>
        <div class="action-row">
          <button type="button" data-preset="pack_hf_openorca" data-stage="tokenize">OpenOrca</button>
          <button type="button" data-preset="pack_hf_tulu3" data-stage="tokenize">Tulu-3 SFT</button>
          <button type="button" data-preset="pack_hf_slimorca" data-stage="tokenize">SlimOrca-Dedup</button>
          <button type="button" data-preset="pack_hf_math" data-stage="tokenize">MetaMathQA</button>
        </div>
      </div>
      <div class="card">
        <h3>Packed shards present</h3>
        <div class="artifact-list" id="artifactListTokenize"></div>
      </div>
    </section>

    <!-- ===== Tab 4: Train ===== -->
    <section class="tab-page" data-tab="train">
      <div class="page-head">
        <h2>Train — distributed run on the packed shards</h2>
        <p class="page-sub">Runs <code>train_3xa100.sh</code> with the session's packed dir as input and the session's
        <code>run/</code> as the checkpoint output. Multi-session checkpoints stay isolated.</p>
      </div>
      <div class="train-grid">
        <div class="card train-controls">
          <h3>Launch</h3>
          <div class="field"><label for="runSteps">Steps Override (0 = full)</label><input id="runSteps" type="number" min="0" value="0"></div>
          <div class="action-row">
            <button class="primary" type="button" data-preset="train_300m_distill" data-stage="train">Start 300M Distill</button>
            <button type="button" data-preset="smoke_300m_distill" data-stage="train">Smoke 1 Step</button>
            <button type="button" data-preset="train_byte_infer" data-stage="train">Train Byte</button>
          </div>
        </div>
        <div class="card train-runs">
          <h3>Runs <span class="muted small" id="runCount">0</span></h3>
          <div class="run-list" id="runList"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-head">
          <h3 id="runTitle">Loss</h3>
          <span class="muted small" id="updatedAt">-</span>
        </div>
        <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
        <div class="stats" id="stats"></div>
        <div class="table-wrap"><table id="metricTable"></table></div>
      </div>
      <div class="card">
        <div class="card-head">
          <h3>Telemetry</h3>
          <span class="muted small" id="telemetryStatus">select a run</span>
        </div>
        <div class="subtabs">
          <button class="subtab-btn active" type="button" data-subtab="gradView">Gradient Heatmap</button>
          <button class="subtab-btn" type="button" data-subtab="lossTableView">Loss Table</button>
        </div>
        <div class="subtab-view active" id="gradView">
          <div class="heatmap-tools">
            <input id="gradStepSlider" type="range" min="0" max="0" value="0">
            <span class="health" id="gradStepLabel">no gradient rows</span>
          </div>
          <canvas id="gradHeatmap"></canvas>
          <div class="grad-detail" id="gradDetail">No gradient telemetry yet. New training runs log per-module gradient norms.</div>
        </div>
        <div class="subtab-view" id="lossTableView">
          <div class="table-wrap"><table id="telemetryTable"></table></div>
        </div>
      </div>
    </section>

    <!-- ===== Tab 5: Evaluate ===== -->
    <section class="tab-page" data-tab="evaluate">
      <div class="page-head">
        <h2>Evaluate — convergence checks &amp; checkpoint chat</h2>
        <p class="page-sub">Runs convergence diagnostics on the active session's metrics and lets you sample any
        checkpoint with a chat prompt.</p>
      </div>
      <div class="cards-2col">
        <div class="card">
          <h3>Diagnostics</h3>
          <div class="grid-2">
            <div class="field"><label for="evalPrompt">Eval Prompt</label><input id="evalPrompt" value="Solve step by step: If 3x + 7 = 31, what is x?"></div>
            <div class="field"><label for="sampleTokens">Sample Tokens</label><input id="sampleTokens" type="number" min="1" max="1024" value="256"></div>
          </div>
          <div class="action-row">
            <button type="button" data-preset="check_convergence_session" data-stage="evaluate">Session Convergence</button>
            <button type="button" data-preset="check_convergence_selected" data-stage="evaluate">Selected Run Convergence</button>
            <button type="button" data-preset="sample_selected_checkpoint" data-stage="evaluate">Sample Selected Ckpt</button>
          </div>
        </div>
        <div class="card chat-card">
          <div class="card-head">
            <h3>Checkpoint Chat</h3>
            <span class="muted small" id="chatStatus">idle</span>
          </div>
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
        </div>
      </div>
    </section>

    <!-- ===== Tab 6: Console (raw command runner) ===== -->
    <section class="tab-page" data-tab="console">
      <div class="page-head">
        <h2>Console — raw command runner &amp; live job log</h2>
        <p class="page-sub">Run any preset directly, watch its output, stop misbehaving jobs.
        This is also where the live log from any stage tab streams.</p>
      </div>
      <div class="console-body">
        <div class="command-form card">
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
            <label>Jobs <span class="muted small" id="commandStatus">idle</span></label>
            <div class="job-list" id="jobList"></div>
          </div>
        </div>
        <div class="terminal card">
          <div class="terminal-head"><strong id="activeJobTitle">No job selected</strong><span class="health" id="activeJobMeta">-</span></div>
          <pre id="commandLog"></pre>
        </div>
      </div>
    </section>
  </main>

  <!-- hidden / unused but referenced by old JS: keep zero-impact placeholders -->
  <div hidden>
    <span id="pipelineSummary"></span>
    <div id="pipelineList"></div>
    <div id="dataStatus"></div>
    <div id="artifactList"></div>
  </div>

<script>
const state = {
  runs: [], metrics: [], checkpoints: [], artifacts: [],
  selectedRun: null, presets: [], jobs: [], selectedJob: null, selectedGradIndex: 0,
  ollamaModels: [], ollamaReachable: null,
  sessions: [], activeSession: 'default', pipeline: null,
};
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
  await refreshSessions().catch(() => {});
  await refreshPipeline().catch(() => {});
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
  const sessionName = state.activeSession || 'default';
  const sessionPaths = state.pipeline?.paths || {};
  const selectedCheckpoint = $('checkpointSelect').value
    || `${sessionPaths.session_run_dir || 'sessions/' + sessionName + '/run'}/ckpt_final.pt`;
  const selectedRun = state.selectedRun || `${sessionName}/run`;
  const selectedRunPath = state.selectedRun
    ? `logs/${state.selectedRun}`
    : (sessionPaths.session_run_dir || `sessions/${sessionName}/run`);
  return {
    teacher: $('teacherModel').value || 'qwen3:4b',
    ollama_host: 'http://127.0.0.1:11434',
    count: $('sampleCount').value || '50000',
    workers: $('syntheticWorkers').value || '1',
    steps: String(steps),
    maybe_steps: steps > 0 ? ` --steps ${steps}` : '',
    selected_run: selectedRun,
    selected_run_path: shellQuote(selectedRunPath),
    selected_checkpoint: shellQuote(selectedCheckpoint),
    eval_prompt: shellQuote($('evalPrompt').value || 'Solve step by step: If 3x + 7 = 31, what is x?'),
    sample_tokens: $('sampleTokens').value || '256',
    session_name: sessionName,
    session_dir: shellQuote(sessionPaths.session_dir || `sessions/${sessionName}`),
    session_data_jsonl: shellQuote(sessionPaths.session_data_jsonl || `sessions/${sessionName}/data/reasoning.jsonl`),
    session_packed_dir: shellQuote(sessionPaths.session_packed_dir || `sessions/${sessionName}/packed`),
    session_run_dir: shellQuote(sessionPaths.session_run_dir || `sessions/${sessionName}/run`),
    session_eval_dir: shellQuote(sessionPaths.session_eval_dir || `sessions/${sessionName}/eval`),
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
$('teacherModel').addEventListener('change', () => persistSessionConfig());
$('sampleCount').addEventListener('change', () => persistSessionConfig());
$('syntheticWorkers').addEventListener('change', () => persistSessionConfig());
$('reloadPresetsBtn').addEventListener('click', () => Promise.all([refreshPresets(), refreshJobs(), refreshArtifacts()]).catch(err => alert(err.message)));
$('refreshDataBtn').addEventListener('click', () => refreshArtifacts().catch(err => alert(err.message)));
$('startJobBtn').addEventListener('click', () => startJob());
$('stopJobBtn').addEventListener('click', stopJob);
$('gradStepSlider').addEventListener('input', () => { state.selectedGradIndex = Number($('gradStepSlider').value || 0); drawGradHeatmap(); });
document.querySelectorAll('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => {
    const stage = btn.dataset.stage;
    if (stage) {
      const blocker = stageBlocker(stage);
      if (blocker) warnStage(stage, blocker);
    }
    startPreset(btn.dataset.preset);
  });
});

// --- theme toggle (persists to localStorage) ---
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const btn = $('themeToggleBtn');
  if (btn) btn.textContent = (theme === 'dark') ? '☼' : '☾';
  try { localStorage.setItem('phi-theme', theme); } catch (_) {}
}
(function initTheme() {
  let theme = 'light';
  try { theme = localStorage.getItem('phi-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); }
  catch (_) {}
  applyTheme(theme);
})();
$('themeToggleBtn').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
});

// --- session selector ---
$('sessionSelect').addEventListener('change', async () => {
  const name = $('sessionSelect').value;
  try {
    await getJSON('/api/sessions/activate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    state.activeSession = name;
    await Promise.all([refreshSessions(), refreshPipeline()]);
    applyPreset();
  } catch (err) { alert(err.message); }
});
$('createSessionBtn').addEventListener('click', async () => {
  const name = ($('newSessionName').value || '').trim();
  if (!name) { alert('Enter a session name.'); return; }
  try {
    await getJSON('/api/sessions/create', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        model: $('teacherModel').value || 'qwen3:4b',
        count: Number($('sampleCount').value || 50000),
        workers: Number($('syntheticWorkers').value || 1),
      })
    });
    await getJSON('/api/sessions/activate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    state.activeSession = name;
    $('newSessionName').value = '';
    await Promise.all([refreshSessions(), refreshPipeline()]);
    applyPreset();
  } catch (err) { alert(err.message); }
});

async function persistSessionConfig() {
  if (!state.activeSession) return;
  try {
    await getJSON('/api/sessions/update', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: state.activeSession,
        model: $('teacherModel').value,
        count: Number($('sampleCount').value || 0) || null,
        workers: Number($('syntheticWorkers').value || 0) || null,
      })
    });
  } catch (_) {}
}

async function refreshSessions() {
  const data = await getJSON('/api/sessions');
  state.sessions = data.sessions || [];
  state.activeSession = data.active || 'default';
  const sel = $('sessionSelect');
  const old = sel.value;
  sel.replaceChildren();
  for (const s of state.sessions) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = `${s.name}${s.model ? ' · ' + s.model : ''}`;
    sel.appendChild(opt);
  }
  sel.value = state.activeSession;
  $('activeSessionLabel').textContent = state.activeSession;
  $('sessionChip').textContent = `session: ${state.activeSession}`;
  // hydrate config inputs from active session
  const active = state.sessions.find(s => s.name === state.activeSession);
  if (active) {
    if (active.model && !document.activeElement?.matches('#teacherModel')) {
      const opt = [...$('teacherModel').options].find(o => o.value === active.model);
      if (opt) $('teacherModel').value = active.model;
    }
    if (active.count && !document.activeElement?.matches('#sampleCount')) {
      $('sampleCount').value = active.count;
    }
    if (active.workers && !document.activeElement?.matches('#syntheticWorkers')) {
      $('syntheticWorkers').value = active.workers;
    }
  }
}

async function refreshOllamaModels() {
  let data;
  try { data = await getJSON('/api/ollama/models'); }
  catch (err) { state.ollamaReachable = false; renderOllamaDropdown([]); return; }
  state.ollamaReachable = !!data.reachable;
  state.ollamaModels = data.models || [];
  renderOllamaDropdown(state.ollamaModels);
}

function renderOllamaDropdown(models) {
  const sel = $('teacherModel');
  const current = sel.value;
  sel.replaceChildren();
  if (!state.ollamaReachable) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = '(Ollama unreachable — start `ollama serve`)';
    sel.appendChild(opt);
    return;
  }
  if (!models.length) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = '(no local models — run `ollama pull <model>`)';
    sel.appendChild(opt);
    return;
  }
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.name;
    const sizeGB = (m.size_mb || 0) / 1024;
    const params = m.details?.parameter_size || '';
    opt.textContent = `${m.name} · ${sizeGB.toFixed(1)} GB${params ? ' · ' + params : ''}`;
    sel.appendChild(opt);
  }
  // restore selection: explicit, then session-config, then first
  const active = state.sessions.find(s => s.name === state.activeSession);
  const preferred = current || active?.model;
  if (preferred && models.some(m => m.name === preferred)) sel.value = preferred;
  applyPreset();
}

async function refreshPipeline() {
  let data;
  try { data = await getJSON('/api/pipeline/state'); }
  catch (err) { return; }
  state.pipeline = data;
  renderPipeline();
  applyGating();
  applyPreset();
}

function renderPipeline() {
  const list = $('pipelineList');
  list.replaceChildren();
  if (!state.pipeline || !state.pipeline.stages) {
    const div = document.createElement('div'); div.className = 'empty'; div.textContent = 'no pipeline state';
    list.appendChild(div); return;
  }
  let doneCount = 0;
  for (const stage of state.pipeline.stages) {
    if (stage.status === 'done') doneCount++;
    const row = document.createElement('div');
    row.className = 'pipeline-item';
    row.setAttribute('data-status', stage.status);
    const dot = document.createElement('div'); dot.className = 'dot';
    const center = document.createElement('div');
    const label = document.createElement('div'); label.className = 'pipeline-label'; label.textContent = stage.label;
    const detail = document.createElement('div'); detail.className = 'pipeline-detail'; detail.textContent = stage.detail || '';
    center.append(label, detail);
    const right = document.createElement('div'); right.className = 'pipeline-step-num';
    right.textContent = stage.status === 'done' ? '✓' : stage.status === 'in_progress' ? '…' : '·';
    row.append(dot, center, right);
    list.appendChild(row);
  }
  $('pipelineSummary').textContent = `${doneCount}/${state.pipeline.stages.length} stages done`;
}

function stageById(id) {
  return state.pipeline?.stages?.find(s => s.id === id) || null;
}

function stageBlocker(id) {
  const stage = stageById(id);
  if (!stage || !stage.blocked_by) return null;
  const upstream = stageById(stage.blocked_by);
  return upstream ? upstream.label : stage.blocked_by;
}

function applyGating() {
  // Soft mode: never block clicks. Mark stage buttons that have an unmet
  // prerequisite so the UI flags them, and update the top tab dots.
  if (!state.pipeline) return;
  const blockedStages = new Set();
  for (const stage of state.pipeline.stages) {
    if (stage.blocked_by) blockedStages.add(stage.id);
  }
  document.querySelectorAll('[data-stage]').forEach(btn => {
    const blocker = stageBlocker(btn.dataset.stage);
    if (blocker) {
      btn.dataset.stageWarn = 'true';
      btn.title = `Heads up: "${blocker}" hasn't produced its artifact yet. This will likely fail.`;
    } else {
      btn.dataset.stageWarn = 'false';
      btn.title = '';
    }
  });
  // Update top-tab dots
  for (const stage of state.pipeline.stages) {
    const pill = document.querySelector(`.tab-pill[data-stage-key="${stage.id}"]`);
    if (!pill) continue;
    const dot = pill.querySelector('.tab-dot');
    if (dot) dot.setAttribute('data-status', stage.status);
  }
  // Setup tab is "done" as soon as a session + teacher exist
  const setupPill = document.querySelector('.tab-pill[data-stage-key="setup"] .tab-dot');
  if (setupPill) {
    const hasTeacher = !!$('teacherModel').value;
    const hasSession = !!state.activeSession;
    setupPill.setAttribute('data-status', (hasSession && hasTeacher) ? 'done' : 'in_progress');
  }
}

let _warningTimer = null;
function showWarning(message, autoHideMs = 7000) {
  const bar = $('warningBar');
  if (!bar) return;
  $('warningBarText').textContent = message;
  bar.hidden = false;
  if (_warningTimer) clearTimeout(_warningTimer);
  if (autoHideMs > 0) {
    _warningTimer = setTimeout(() => { bar.hidden = true; }, autoHideMs);
  }
}
function warnStage(stage, blocker) {
  showWarning(`Stage "${stage}" needs "${blocker}" first. The command will still run, but expect it to fail until the previous artifact exists.`);
}

function setActiveTab(name) {
  const valid = ['setup','generate','tokenize','train','evaluate','console'];
  if (!valid.includes(name)) name = 'setup';
  document.querySelectorAll('.tab-page').forEach(page => {
    page.setAttribute('data-active', String(page.dataset.tab === name));
  });
  document.querySelectorAll('.tab-pill').forEach(pill => {
    pill.setAttribute('aria-current', String(pill.dataset.tab === name));
  });
  state.activeTab = name;
  try { localStorage.setItem('phi-tab', name); } catch (_) {}
  // Tab-specific refresh hooks
  if (name === 'train' || name === 'evaluate') {
    drawChart(); drawGradHeatmap();
  }
}

function pickInitialTab() {
  try {
    const saved = localStorage.getItem('phi-tab');
    if (saved) { setActiveTab(saved); return; }
  } catch (_) {}
  // Auto-advance to the first non-done stage
  if (state.pipeline?.stages) {
    const first = state.pipeline.stages.find(s => s.status !== 'done');
    if (first) { setActiveTab(first.id); return; }
  }
  setActiveTab('setup');
}

document.querySelectorAll('.tab-pill').forEach(pill => {
  pill.addEventListener('click', () => setActiveTab(pill.dataset.tab));
});
const _warnClose = $('warningBarClose');
if (_warnClose) _warnClose.addEventListener('click', () => { $('warningBar').hidden = true; });

// Subtabs inside the Train > Telemetry card mirror the previous behavior.
document.querySelectorAll('[data-subtab]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.subtab-btn').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.subtab-view').forEach(view => view.classList.toggle('active', view.id === btn.dataset.subtab));
    drawGradHeatmap();
  });
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
refreshOllamaModels().catch(() => {});
setInterval(() => refreshAll().catch(() => {}), 5000);
setInterval(() => refreshJobs().catch(() => {}), 2000);
setInterval(() => refreshArtifacts().catch(() => {}), 10000);
setInterval(() => refreshPipeline().catch(() => {}), 4000);
setInterval(() => refreshOllamaModels().catch(() => {}), 30000);
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


_PYTHON_TOKEN_RE = re.compile(r"(?<![\w./-])python(?![\w./-])")


COMMAND_PRESETS = [
    # --- Stage 1: synthetic data generation (session-isolated) ---
    {
        "id": "generate_synthetic",
        "label": "[1] Generate reasoning JSONL (session)",
        "stage": "generate",
        "command": "PYTHONPATH=src python scripts/generate_synthetic_reasoning.py --model {teacher} --host {ollama_host} --seed-prompts prompts/reasoning_seeds.jsonl --out {session_data_jsonl} --count {count} --workers {workers} --heartbeat-every 5",
    },
    {
        "id": "generate_synthetic_smoke",
        "label": "[1] Generate 20 smoke records (session)",
        "stage": "generate",
        "command": "PYTHONPATH=src python scripts/generate_synthetic_reasoning.py --model {teacher} --host {ollama_host} --seed-prompts prompts/reasoning_seeds.jsonl --out {session_data_jsonl} --count 20 --workers 1 --heartbeat-every 5",
    },
    # --- Stage 2: tokenization / packing ---
    {
        "id": "pack_session_qwen",
        "label": "[2] Pack session JSONL (Qwen tokenizer)",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --input {session_data_jsonl} --tokenizer Qwen/Qwen2.5-1.5B --out {session_packed_dir} --text-column text --validation-every 100",
    },
    {
        "id": "pack_session_byte",
        "label": "[2] Pack session JSONL (byte tokenizer)",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --input {session_data_jsonl} --tokenizer byte --out {session_packed_dir} --text-column text --validation-every 100",
    },
    {
        "id": "pack_hf_openorca",
        "label": "[2] Pack OpenOrca (HF) → session",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --hf-dataset Open-Orca/OpenOrca --hf-split train --text-column response --tokenizer Qwen/Qwen2.5-1.5B --out {session_packed_dir} --max-docs 50000 --validation-every 200",
    },
    {
        "id": "pack_hf_tulu3",
        "label": "[2] Pack Tulu-3 SFT (HF) → session",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --hf-dataset allenai/tulu-3-sft-mixture --hf-split train --text-column messages --tokenizer Qwen/Qwen2.5-1.5B --out {session_packed_dir} --max-docs 30000 --validation-every 200",
    },
    {
        "id": "pack_hf_slimorca",
        "label": "[2] Pack SlimOrca-Dedup (HF) → session",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --hf-dataset Open-Orca/SlimOrca-Dedup --hf-split train --text-column conversations --tokenizer Qwen/Qwen2.5-1.5B --out {session_packed_dir} --max-docs 30000 --validation-every 200",
    },
    {
        "id": "pack_hf_math",
        "label": "[2] Pack MetaMathQA (HF) → session",
        "stage": "tokenize",
        "command": "PYTHONPATH=src python scripts/build_token_shards.py --hf-dataset meta-math/MetaMathQA --hf-split train --text-column response --tokenizer Qwen/Qwen2.5-1.5B --out {session_packed_dir} --max-docs 20000 --validation-every 200",
    },
    # --- Stage 3: training (session-isolated checkpoints) ---
    {
        "id": "train_300m_distill",
        "label": "[3] Train 300M distill (session)",
        "stage": "train",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml --out-dir {session_run_dir} --packed-dir {session_packed_dir} --run-name {session_name}{maybe_steps}",
    },
    {
        "id": "smoke_300m_distill",
        "label": "[3] Smoke train 1 step (session)",
        "stage": "train",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_300m_qwen_distill.yaml --out-dir {session_run_dir} --packed-dir {session_packed_dir} --run-name {session_name} --steps 1",
    },
    {
        "id": "train_byte_infer",
        "label": "[3] Train byte inference (session)",
        "stage": "train",
        "command": "bash scripts/train_3xa100.sh configs/a100_3gpu_plasma_byte_infer.yaml --out-dir {session_run_dir} --packed-dir {session_packed_dir} --run-name {session_name}{maybe_steps}",
    },
    # --- Stage 4: evaluation ---
    {
        "id": "check_convergence_selected",
        "label": "[4] Check convergence (selected run)",
        "stage": "evaluate",
        "command": "PYTHONPATH=src python scripts/convergence_check.py --run {selected_run_path}",
    },
    {
        "id": "sample_selected_checkpoint",
        "label": "[4] Sample from selected checkpoint",
        "stage": "evaluate",
        "command": "PYTHONPATH=src python scripts/generate.py --ckpt {selected_checkpoint} --prompt {eval_prompt} --device cuda --max-new-tokens {sample_tokens}",
    },
    {
        "id": "check_convergence_session",
        "label": "[4] Check convergence (active session)",
        "stage": "evaluate",
        "command": "PYTHONPATH=src python scripts/convergence_check.py --run {session_run_dir}",
    },
    # --- Utilities ---
    {
        "id": "install_scale_deps",
        "label": "[utility] Install scale dependencies",
        "stage": "setup",
        "command": "python -m pip install --progress-bar on -e '.[scale]'",
    },
    {
        "id": "ollama_models",
        "label": "[utility] List Ollama models",
        "stage": "setup",
        "command": "ollama list",
    },
    {
        "id": "test_suite",
        "label": "[utility] Run tests",
        "stage": "setup",
        "command": "PYTHONPATH=src pytest tests/",
    },
]


_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def _safe_session_name(name: str) -> str:
    name = (name or "").strip()
    if not _SESSION_NAME_RE.match(name):
        raise ValueError(
            "session name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,62} "
            "(letters, digits, underscore, dot, dash; start alphanumeric)"
        )
    return name


class SessionManager:
    """Per-experiment workspace: data, packed shards, checkpoints, config.

    Layout under <root>/sessions/<name>/:
        config.json   — session metadata (model, count, created_at)
        data/         — synthetic JSONL outputs
        packed/       — packed token shards (meta.json marks readiness)
        run/          — training output (metrics.jsonl + ckpt_*.pt)
        eval/         — evaluation artifacts
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.base = self.root / "sessions"
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        active_file = self.base / "active.txt"
        if not active_file.exists():
            active_file.write_text("default\n", encoding="utf-8")
        try:
            self.ensure("default", model="qwen3:4b")
        except ValueError:
            pass

    def session_dir(self, name: str) -> Path:
        name = _safe_session_name(name)
        return self.base / name

    def paths(self, name: str) -> dict[str, str]:
        d = self.session_dir(name)
        return {
            "session_name": name,
            "session_dir": str(d),
            "session_data_jsonl": str(d / "data" / "reasoning.jsonl"),
            "session_packed_dir": str(d / "packed"),
            "session_run_dir": str(d / "run"),
            "session_eval_dir": str(d / "eval"),
            "session_config": str(d / "config.json"),
        }

    def ensure(self, name: str, model: str = "qwen3:4b") -> dict[str, Any]:
        d = self.session_dir(name)
        with self._lock:
            (d / "data").mkdir(parents=True, exist_ok=True)
            (d / "packed").mkdir(parents=True, exist_ok=True)
            (d / "run").mkdir(parents=True, exist_ok=True)
            (d / "eval").mkdir(parents=True, exist_ok=True)
            cfg_path = d / "config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cfg = {}
            else:
                cfg = {}
            cfg.setdefault("name", name)
            cfg.setdefault("created_at", time.time())
            cfg.setdefault("model", model)
            cfg.setdefault("count", 50000)
            cfg.setdefault("workers", 1)
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return cfg

    def update(self, name: str, **fields: Any) -> dict[str, Any]:
        cfg = self.ensure(name)
        cfg.update({k: v for k, v in fields.items() if v is not None})
        cfg["updated_at"] = time.time()
        (self.session_dir(name) / "config.json").write_text(
            json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return cfg

    def active(self) -> str:
        active_file = self.base / "active.txt"
        try:
            name = active_file.read_text(encoding="utf-8").strip() or "default"
        except OSError:
            name = "default"
        try:
            return _safe_session_name(name)
        except ValueError:
            return "default"

    def set_active(self, name: str) -> str:
        name = _safe_session_name(name)
        self.ensure(name)
        (self.base / "active.txt").write_text(name + "\n", encoding="utf-8")
        return name

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.base.iterdir()):
            if not path.is_dir():
                continue
            try:
                _safe_session_name(path.name)
            except ValueError:
                continue
            cfg_path = path / "config.json"
            cfg: dict[str, Any] = {}
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cfg = {}
            result.append({
                "name": path.name,
                "model": cfg.get("model"),
                "count": cfg.get("count"),
                "workers": cfg.get("workers"),
                "created_at": cfg.get("created_at"),
                "updated_at": cfg.get("updated_at"),
            })
        return result


def fetch_ollama_models(host: str = "http://127.0.0.1:11434", timeout: float = 3.0) -> dict[str, Any]:
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
    url = host.rstrip("/") + "/api/tags"
    try:
        with _urllib_request.urlopen(url, timeout=timeout) as response:
            obj = json.loads(response.read().decode("utf-8"))
    except (_urllib_error.URLError, TimeoutError, ValueError) as exc:
        return {"reachable": False, "host": host, "error": str(exc), "models": []}
    models = []
    for m in obj.get("models", []):
        models.append({
            "name": m.get("name"),
            "size_mb": (m.get("size", 0) or 0) / (1024 * 1024),
            "modified_at": m.get("modified_at"),
            "details": {
                "family": (m.get("details", {}) or {}).get("family"),
                "parameter_size": (m.get("details", {}) or {}).get("parameter_size"),
                "quantization": (m.get("details", {}) or {}).get("quantization_level"),
            },
        })
    models.sort(key=lambda x: (x.get("size_mb") or 0))
    return {"reachable": True, "host": host, "models": models}


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
        self._venv_lock = threading.Lock()
        self._python_path: str | None = None

    def presets(self) -> list[dict[str, str]]:
        return list(COMMAND_PRESETS)

    def python_executable(self) -> str:
        if self._python_path is not None:
            return self._python_path
        with self._venv_lock:
            if self._python_path is not None:
                return self._python_path
            venv_dir = self.root / ".venv"
            venv_python = venv_dir / "bin" / "python"
            if not venv_python.exists():
                print(f"[dashboard] creating venv at {venv_dir} (one-time, ~15s)…", flush=True)
                try:
                    subprocess.run(
                        [sys.executable, "-m", "venv", str(venv_dir)],
                        check=True,
                        cwd=str(self.root),
                    )
                    subprocess.run(
                        [str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"],
                        check=False,
                        cwd=str(self.root),
                    )
                    print(f"[dashboard] venv ready: {venv_python}", flush=True)
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    print(f"[dashboard] venv bootstrap failed ({exc}); falling back to {sys.executable}", flush=True)
                    self._python_path = sys.executable
                    return self._python_path
            self._python_path = str(venv_python)
            return self._python_path

    def start(self, command: str, label: str = "custom") -> CommandJob:
        command = command.strip()
        if not command:
            raise ValueError("command is empty")
        command = _PYTHON_TOKEN_RE.sub(self.python_executable(), command)
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
        self.sessions = SessionManager(self.root)
        threading.Thread(target=self.commands.python_executable, daemon=True).start()

    def pipeline_state(self, session: str | None = None) -> dict[str, Any]:
        name = session or self.sessions.active()
        try:
            paths = self.sessions.paths(name)
        except ValueError as exc:
            return {"error": str(exc)}
        cfg_path = Path(paths["session_config"])
        cfg: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cfg = {}

        jsonl = Path(paths["session_data_jsonl"])
        packed = Path(paths["session_packed_dir"])
        run = Path(paths["session_run_dir"])
        meta_json = packed / "meta.json"

        # generate
        jsonl_records = count_lines(jsonl) if jsonl.exists() else 0
        jsonl_size = jsonl.stat().st_size if jsonl.exists() else 0
        progress_path = jsonl.with_suffix(jsonl.suffix + ".progress.json")
        progress: dict[str, Any] = {}
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}

        # tokenize
        packed_meta: dict[str, Any] = {}
        tokens_total = 0
        if meta_json.exists():
            try:
                packed_meta = json.loads(meta_json.read_text(encoding="utf-8"))
                splits = packed_meta.get("splits") or {}
                tokens_total = int((splits.get("train") or {}).get("tokens") or 0) \
                    + int((splits.get("validation") or {}).get("tokens") or 0)
            except (OSError, json.JSONDecodeError):
                packed_meta = {}

        # train
        ckpts: list[dict[str, Any]] = []
        metrics_count = 0
        latest_step = 0
        if run.exists():
            metrics_path = run / "metrics.jsonl"
            if metrics_path.exists():
                rows = read_jsonl(metrics_path)
                metrics_count = len(rows)
                if rows:
                    latest_step = int(rows[-1].get("step") or 0)
            for p in sorted(run.glob("ckpt_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                ckpts.append({
                    "name": p.name,
                    "path": self.rel(p),
                    "step": parse_step_from_name(p),
                    "size_mb": p.stat().st_size / (1024 * 1024),
                    "updated_at": p.stat().st_mtime,
                })

        def status_for(condition: bool, partial: bool = False) -> str:
            if condition:
                return "done"
            if partial:
                return "in_progress"
            return "pending"

        stages = [
            {
                "id": "generate",
                "label": "1. Synthetic data",
                "status": status_for(jsonl_records >= max(1, int(cfg.get("count", 50000) // 4)),
                                     partial=jsonl_records > 0),
                "detail": (f"{jsonl_records} records | "
                           f"{jsonl_size/1024/1024:.1f} MB" if jsonl_records else "no records yet"),
                "artifact": self.rel(jsonl) if jsonl.exists() else paths["session_data_jsonl"],
                "exists": jsonl.exists(),
                "records": jsonl_records,
                "progress": progress,
            },
            {
                "id": "tokenize",
                "label": "2. Tokenize / pack",
                "status": status_for(meta_json.exists() and tokens_total > 0),
                "detail": (f"{tokens_total} tokens | tokenizer "
                           f"{packed_meta.get('tokenizer_name') or 'byte'}" if meta_json.exists()
                           else "no packed shards"),
                "artifact": self.rel(packed) if packed.exists() else paths["session_packed_dir"],
                "exists": meta_json.exists(),
                "tokens": tokens_total,
                "blocked_by": "generate" if jsonl_records == 0 else None,
            },
            {
                "id": "train",
                "label": "3. Train (DDP)",
                "status": status_for(bool(ckpts), partial=metrics_count > 0),
                "detail": (f"{len(ckpts)} ckpts | last step {latest_step}" if metrics_count
                           else "no training started"),
                "artifact": self.rel(run) if run.exists() else paths["session_run_dir"],
                "exists": run.exists() and metrics_count > 0,
                "metrics_count": metrics_count,
                "latest_step": latest_step,
                "ckpts": len(ckpts),
                "blocked_by": "tokenize" if not meta_json.exists() else None,
            },
            {
                "id": "evaluate",
                "label": "4. Evaluate",
                "status": status_for(False, partial=bool(ckpts)),
                "detail": (f"{len(ckpts)} checkpoints available" if ckpts
                           else "train at least one checkpoint first"),
                "artifact": self.rel(run / "eval") if (run / "eval").exists()
                             else paths["session_eval_dir"],
                "exists": False,
                "blocked_by": "train" if not ckpts else None,
            },
        ]
        return {
            "session": name,
            "config": cfg,
            "paths": paths,
            "stages": stages,
            "checkpoints": ckpts,
        }

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def run_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        if self.logs_dir.exists():
            dirs.extend(p for p in self.logs_dir.iterdir()
                        if p.is_dir() and (p / "metrics.jsonl").exists())
        sessions_dir = self.root / "sessions"
        if sessions_dir.exists():
            for session in sessions_dir.iterdir():
                run_dir = session / "run"
                if run_dir.is_dir() and (run_dir / "metrics.jsonl").exists():
                    dirs.append(run_dir)
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
            # Session runs live at sessions/<name>/run; their dir name is "run" which collides.
            # Use a path-like name so the Runs panel disambiguates them.
            try:
                if run_dir.parent.parent == (self.root / "sessions"):
                    name = f"{run_dir.parent.name}/run"
                else:
                    name = run_dir.name
            except OSError:
                name = run_dir.name
            result.append({
                "name": name,
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
        # Accept logs/<run> names AND sessions/<name>/run synthetic names ("<name>/run").
        if "/" in run_name:
            run_dir = (self.root / "sessions" / run_name).resolve()
            sessions_root = (self.root / "sessions").resolve()
            if sessions_root not in run_dir.parents:
                raise ValueError("run path escapes sessions dir")
        else:
            run_dir = (self.logs_dir / run_name).resolve()
            if run_dir != self.logs_dir and self.logs_dir not in run_dir.parents:
                raise ValueError("run path escapes logs dir")
        return read_jsonl(run_dir / "metrics.jsonl", limit=limit)

    def checkpoints(self) -> list[dict[str, Any]]:
        rows = []
        ckpt_paths: list[Path] = []
        if self.logs_dir.exists():
            ckpt_paths.extend(self.logs_dir.rglob("*.pt"))
        sessions_dir = self.root / "sessions"
        if sessions_dir.exists():
            ckpt_paths.extend(sessions_dir.glob("*/run/*.pt"))
        for path in sorted(ckpt_paths, key=lambda p: p.stat().st_mtime, reverse=True):
            run_label = path.parent.name
            try:
                if path.parent.parent.parent == sessions_dir:
                    run_label = f"{path.parent.parent.name}/run"
            except OSError:
                pass
            rows.append({
                "name": path.name,
                "path": self.rel(path),
                "run": run_label,
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

        sessions_dir = self.root / "sessions"
        if sessions_dir.exists():
            for sess_jsonl in sorted(sessions_dir.glob("*/data/*.jsonl"),
                                     key=lambda p: p.stat().st_mtime, reverse=True):
                records = count_lines(sess_jsonl)
                add_file("session-jsonl", sess_jsonl,
                         summary=f"{records} records | session {sess_jsonl.parent.parent.name}",
                         records=records)

        packed_meta_paths: list[Path] = []
        packed_dir = self.root / "data" / "packed"
        if packed_dir.exists():
            packed_meta_paths.extend(packed_dir.rglob("meta.json"))
        if sessions_dir.exists():
            packed_meta_paths.extend(sessions_dir.glob("*/packed/meta.json"))
        for meta_path in sorted(packed_meta_paths, key=lambda p: p.stat().st_mtime, reverse=True):
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
            if parsed.path == "/api/ollama/models":
                self.require_local_command_client()
                query = parse_qs(parsed.query)
                host = query.get("host", ["http://127.0.0.1:11434"])[0]
                self.send_json(fetch_ollama_models(host))
                return
            if parsed.path == "/api/sessions":
                self.require_local_command_client()
                self.send_json({
                    "sessions": self.state.sessions.list(),
                    "active": self.state.sessions.active(),
                })
                return
            if parsed.path == "/api/pipeline/state":
                self.require_local_command_client()
                query = parse_qs(parsed.query)
                name = query.get("session", [self.state.sessions.active()])[0]
                self.send_json(self.state.pipeline_state(name))
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
            if parsed.path == "/api/sessions/create":
                self.require_local_command_client()
                body = self.read_json()
                name = _safe_session_name(str(body.get("name", "")))
                cfg = self.state.sessions.ensure(name, model=str(body.get("model") or "qwen3:4b"))
                self.state.sessions.update(name,
                    model=body.get("model"),
                    count=body.get("count"),
                    workers=body.get("workers"),
                )
                self.send_json({"session": cfg, "paths": self.state.sessions.paths(name)})
                return
            if parsed.path == "/api/sessions/activate":
                self.require_local_command_client()
                body = self.read_json()
                name = self.state.sessions.set_active(str(body.get("name", "")))
                self.send_json({"active": name})
                return
            if parsed.path == "/api/sessions/update":
                self.require_local_command_client()
                body = self.read_json()
                name = _safe_session_name(str(body.get("name") or self.state.sessions.active()))
                cfg = self.state.sessions.update(name,
                    model=body.get("model"),
                    count=body.get("count"),
                    workers=body.get("workers"),
                )
                self.send_json({"session": cfg})
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
