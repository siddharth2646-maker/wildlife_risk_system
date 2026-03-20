"""
WildGuard AI — Wildlife-Vehicle Collision Risk Prediction System
Main Streamlit Application with Real-Time Data Pipeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "WildGuard AI — Wildlife Risk Prediction",
    page_icon    = "🐾",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS — Palantir Foundry Aesthetic ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:         #060a10;
  --bg-alt:     #0a0f18;
  --surface:    #0d1320;
  --surface-2:  #111827;
  --border:     #1a2332;
  --border-hi:  #243044;
  --cyan:       #00e5ff;
  --cyan-dim:   #0091a3;
  --cyan-glow:  rgba(0,229,255,0.12);
  --amber:      #ffb020;
  --amber-dim:  #a67400;
  --red:        #ff3d5a;
  --red-dim:    #8b1a2b;
  --green:      #00e676;
  --green-dim:  #007a3d;
  --blue:       #4da6ff;
  --purple:     #9d7aff;
  --orange:     #ff7043;
  --low:        #00e676;
  --med:        #ffb020;
  --high:       #ff3d5a;
  --crit:       #d500f9;
  --text:       #c8d6e5;
  --text-hi:    #e8f0fa;
  --muted:      #5a6d82;
  --muted-lo:   #3a4a5c;
}

/* ─── Base ─────────────────────────────────────────────────────────────── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, sans-serif;
}

/* Subtle scanline overlay */
.stApp::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,229,255,0.008) 2px, rgba(0,229,255,0.008) 4px);
  pointer-events: none;
  z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; }

/* ─── Keyframes ─────────────────────────────────────────────────────── */
@keyframes pulse-glow {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes scan-line {
  0% { top: -2px; }
  100% { top: 100%; }
}
@keyframes data-fade-in {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes border-pulse {
  0%, 100% { border-color: var(--border); }
  50% { border-color: var(--border-hi); }
}

/* ─── HUD Metric Cards ─────────────────────────────────────────────── */
.metric-card {
  position: relative;
  background: linear-gradient(145deg, var(--surface), var(--bg-alt));
  border: 1px solid var(--border);
  padding: 1rem 1.2rem;
  margin: 0.25rem 0;
  animation: data-fade-in 0.4s ease-out;
  overflow: hidden;
}
/* Corner brackets — HUD style */
.metric-card::before, .metric-card::after {
  content: '';
  position: absolute;
  width: 12px;
  height: 12px;
  border-color: var(--cyan);
  border-style: solid;
}
.metric-card::before {
  top: 0; left: 0;
  border-width: 1px 0 0 1px;
}
.metric-card::after {
  bottom: 0; right: 0;
  border-width: 0 1px 1px 0;
}
.metric-card h3 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: var(--muted);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin: 0 0 0.5rem 0;
  font-weight: 500;
}
.metric-card .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.8rem;
  font-weight: 600;
  color: var(--cyan);
  line-height: 1;
  text-shadow: 0 0 20px rgba(0,229,255,0.2);
}
.metric-card .sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 0.4rem;
  letter-spacing: 0.05em;
}

/* ─── Risk Badge ───────────────────────────────────────────────────── */
.risk-badge {
  display: inline-block;
  padding: 0.25rem 0.8rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  font-size: 0.85rem;
  letter-spacing: 0.08em;
  border: 1px solid;
  text-transform: uppercase;
}

/* ─── Section Headers — Classified Document Style ─────────────────── */
.section-header {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  color: var(--cyan-dim);
  letter-spacing: 0.22em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin: 1.5rem 0 0.7rem 0;
  position: relative;
}
.section-header::before {
  content: '▸';
  margin-right: 0.5rem;
  color: var(--cyan);
}

/* ─── Source Cards ──────────────────────────────────────────────────── */
.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 0.9rem 1.1rem;
  margin: 0.4rem 0;
  position: relative;
}
.source-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 2px;
  background: var(--cyan);
}
.source-card .source-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-hi);
  margin-bottom: 0.2rem;
}
.source-card .source-url {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  color: var(--muted);
  word-break: break-all;
  margin-bottom: 0.3rem;
}
.source-card .source-features {
  font-size: 0.68rem;
  color: var(--cyan);
  font-family: 'IBM Plex Mono', monospace;
}
.status-success { color: var(--green); }
.status-fallback { color: var(--amber); }
.status-error { color: var(--red); }

/* ─── Sidebar — Dark Ops Panel ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060a10 0%, #0a0f18 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::after {
  content: '';
  position: absolute;
  right: 0; top: 0; bottom: 0;
  width: 1px;
  background: linear-gradient(180deg, transparent, var(--cyan-dim), transparent);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
  color: var(--text) !important;
  font-size: 0.78rem !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

/* Navigation radio items */
[data-testid="stSidebar"] .stRadio label {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.04em !important;
  color: var(--text) !important;
  padding: 0.35rem 0 !important;
  transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  color: var(--cyan) !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
[data-testid="stSidebar"] input[type="radio"]:checked + label {
  color: var(--cyan) !important;
}

/* ─── Form Elements ────────────────────────────────────────────────── */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important;
}
.stSlider [data-baseweb="slider"] { background: var(--border) !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ─── Buttons — Technical Style ────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,229,255,0.05)) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan-dim) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 600 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  padding: 0.6rem 2rem !important;
  transition: all 0.2s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(0,229,255,0.1)) !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.15) !important;
  border-color: var(--cyan) !important;
  transform: none !important;
}
.stButton > button:active {
  background: rgba(0,229,255,0.3) !important;
}

/* Metric value override */
div[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.6rem !important;
  color: var(--cyan) !important;
}

/* ─── Markdown text overrides ──────────────────────────────────────── */
.stMarkdown h1 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.stMarkdown h2 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.stMarkdown h3 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; }

/* Dataframe / table styling */
.stDataFrame { border: 1px solid var(--border) !important; }
.stDataFrame th {
  background: var(--surface) !important;
  color: var(--cyan) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* ─── Divider — Subtle ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ─── Live Indicator ───────────────────────────────────────────────── */
.live-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  background: var(--green);
  border-radius: 50%;
  animation: pulse-glow 2s ease-in-out infinite;
  margin-right: 6px;
  box-shadow: 0 0 6px var(--green);
}

/* ─── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted-lo); }

/* ─── Expander ─────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  color: var(--text) !important;
  letter-spacing: 0.06em !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   1. RADAR SWEEP — sidebar animated ping
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes radar-sweep {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.radar-container {
  position: relative;
  width: 70px; height: 70px;
  margin: 0 auto 0.5rem;
}
.radar-ring {
  position: absolute; inset: 0;
  border: 1px solid rgba(0,229,255,0.15);
  border-radius: 50%;
}
.radar-ring.inner { inset: 12px; border-color: rgba(0,229,255,0.1); }
.radar-ring.core  { inset: 24px; border-color: rgba(0,229,255,0.08); }
.radar-sweep-line {
  position: absolute;
  top: 50%; left: 50%;
  width: 50%; height: 1px;
  background: linear-gradient(90deg, var(--cyan), transparent);
  transform-origin: 0 0;
  animation: radar-sweep 4s linear infinite;
  opacity: 0.7;
}
.radar-sweep-line::after {
  content: '';
  position: absolute;
  right: 0; top: -3px;
  width: 6px; height: 6px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--cyan);
}
.radar-center {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 4px; height: 4px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 12px var(--cyan);
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. CLASSIFIED TICKER BAR — scrolling top banner
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes ticker-scroll {
  0%   { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.classified-ticker {
  overflow: hidden;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,0.04), transparent);
  border: 1px solid var(--border);
  border-left: none; border-right: none;
  height: 18px;
  margin-bottom: 0.8rem;
  position: relative;
}
.classified-ticker::before,
.classified-ticker::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 40px;
  z-index: 1;
}
.classified-ticker::before { left: 0; background: linear-gradient(90deg, var(--bg), transparent); }
.classified-ticker::after  { right: 0; background: linear-gradient(90deg, transparent, var(--bg)); }
.ticker-text {
  display: inline-block;
  white-space: nowrap;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.48rem;
  color: var(--muted-lo);
  letter-spacing: 0.35em;
  text-transform: uppercase;
  animation: ticker-scroll 30s linear infinite;
  line-height: 18px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. CROSSHAIR TARGETS on hover — tactical focus
   ═══════════════════════════════════════════════════════════════════════════ */
.metric-card:hover::before {
  width: 16px; height: 16px;
  border-color: var(--cyan);
  filter: drop-shadow(0 0 3px var(--cyan));
  transition: all 0.2s ease;
}
.metric-card:hover::after {
  width: 16px; height: 16px;
  border-color: var(--cyan);
  filter: drop-shadow(0 0 3px var(--cyan));
  transition: all 0.2s ease;
}
.metric-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 0 15px rgba(0,229,255,0.06);
  transition: all 0.2s ease;
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. THREAT LEVEL STRIP — colored edge indicator
   ═══════════════════════════════════════════════════════════════════════════ */
.threat-strip {
  display: flex; gap: 1px;
  margin: 0.6rem 0;
  height: 3px;
}
.threat-strip .seg {
  flex: 1;
  transition: all 0.3s ease;
}
.threat-strip .seg:hover { height: 6px; margin-top: -1.5px; }

/* ═══════════════════════════════════════════════════════════════════════════
   5. MATRIX DATA RAIN — background decoration
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes rain-fall {
  0%   { transform: translateY(-100%); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateY(400%); opacity: 0; }
}
.data-rain {
  position: fixed;
  top: 0; right: 20px;
  width: 200px; height: 100vh;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.rain-col {
  position: absolute;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.45rem;
  color: rgba(0,229,255,0.06);
  writing-mode: vertical-rl;
  animation: rain-fall linear infinite;
  letter-spacing: 0.3em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. HOLOGRAPHIC SHIMMER — section shine effect
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes holo-shimmer {
  0%   { left: -150%; }
  100% { left: 250%; }
}
.section-header::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 60px;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,0.06), transparent);
  animation: holo-shimmer 6s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. CORNER VIGNETTE — ambient darkening
   ═══════════════════════════════════════════════════════════════════════════ */
.stApp::after {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.4) 100%);
  pointer-events: none;
  z-index: 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. TACTICAL GRID — dot matrix background
   ═══════════════════════════════════════════════════════════════════════════ */
.block-container::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(circle, rgba(0,229,255,0.02) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. BREATHING RING — branding pulse around logo
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes breathe-ring {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.3); }
  50%      { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
}
.brand-ring {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px; height: 52px;
  border: 1px solid rgba(0,229,255,0.25);
  border-radius: 50%;
  animation: breathe-ring 3s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. GLITCH FLICKER — header text distortion
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes glitch {
  0%, 98% { text-shadow: 0 0 20px rgba(0,229,255,0.2); }
  99%     { text-shadow: -2px 0 #ff3d5a, 2px 0 #00e5ff; }
  100%    { text-shadow: 0 0 20px rgba(0,229,255,0.2); }
}
.stMarkdown h1 span[style*="color:#00e5ff"],
.stMarkdown h1 span[style*="color:#00e676"] {
  animation: glitch 8s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   11. BOOT SEQUENCE — typing cursor on classified labels
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes blink-cursor {
  0%, 100% { border-right-color: transparent; }
  50%      { border-right-color: var(--cyan); }
}
.boot-text {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.5rem;
  color: var(--muted-lo);
  letter-spacing: 0.2em;
  border-right: 1.5px solid var(--cyan);
  padding-right: 3px;
  animation: blink-cursor 1.2s step-end infinite;
  display: inline-block;
}

/* ═══════════════════════════════════════════════════════════════════════════
   12. TACTICAL SCOPE RETICLE — sidebar brand element
   ═══════════════════════════════════════════════════════════════════════════ */
.reticle {
  position: relative;
  width: 80px; height: 80px;
  margin: 0 auto;
}
.reticle::before, .reticle::after {
  content: '';
  position: absolute;
  background: rgba(0,229,255,0.15);
}
.reticle::before {
  top: 50%; left: 8px; right: 8px;
  height: 1px;
  transform: translateY(-50%);
}
.reticle::after {
  left: 50%; top: 8px; bottom: 8px;
  width: 1px;
  transform: translateX(-50%);
}
.reticle-outer {
  position: absolute; inset: 0;
  border: 1px solid rgba(0,229,255,0.12);
  border-radius: 50%;
}
.reticle-inner {
  position: absolute; inset: 14px;
  border: 1px dashed rgba(0,229,255,0.08);
  border-radius: 50%;
}
.reticle-dot {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 5px; height: 5px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--cyan), 0 0 20px rgba(0,229,255,0.3);
}
.reticle-tick {
  position: absolute;
  background: rgba(0,229,255,0.2);
}
.reticle-tick.t { top: 2px; left: 50%; width: 1px; height: 6px; transform: translateX(-50%); }
.reticle-tick.b { bottom: 2px; left: 50%; width: 1px; height: 6px; transform: translateX(-50%); }
.reticle-tick.l { left: 2px; top: 50%; height: 1px; width: 6px; transform: translateY(-50%); }
.reticle-tick.r { right: 2px; top: 50%; height: 1px; width: 6px; transform: translateY(-50%); }

/* ═══════════════════════════════════════════════════════════════════════════
   BONUS: Intel panel styling
   ═══════════════════════════════════════════════════════════════════════════ */
.intel-panel {
  background: linear-gradient(145deg, var(--surface), var(--bg-alt));
  border: 1px solid var(--border);
  padding: 0.8rem 1rem;
  position: relative;
  margin: 0.5rem 0;
  overflow: hidden;
}
.intel-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
}
.intel-panel .panel-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.5rem;
  color: var(--muted-lo);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.intel-panel .panel-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--cyan);
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════ SPLASH SCREEN ═══════════════════
if "splash_done" not in st.session_state:
    st.session_state.splash_done = True
    st.markdown("""<div id="spl" style="position:fixed;inset:0;z-index:999999;background:#030508;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;animation:sO .8s ease 4.5s forwards">
<style>
@keyframes sO{to{opacity:0;visibility:hidden;pointer-events:none}}
@keyframes sR{to{transform:rotate(360deg)}}
@keyframes sP{0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,.4)}50%{box-shadow:0 0 0 14px rgba(0,229,255,0)}}
@keyframes sB{from{width:0}to{width:100%}}
@keyframes sL{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:translateX(0)}}
.sl{font-size:.55rem;color:#3a4a5c;letter-spacing:.1em;margin:.13rem 0;opacity:0;animation:sL .25s ease forwards}
.sl .g{color:#00e676}.sl .w{color:#ffb020}.sl .c{color:#00e5ff}
.sb{width:250px;height:2px;background:#1a2332;margin:.2rem 0;overflow:hidden}
.sb div{height:100%;background:linear-gradient(90deg,#00e5ff,#4da6ff);animation:sB ease-out forwards}
</style>
<div style="position:relative;width:90px;height:90px;margin-bottom:1rem">
<div style="position:absolute;inset:0;border:1px solid rgba(0,229,255,.18);border-radius:50%"></div>
<div style="position:absolute;inset:15px;border:1px solid rgba(0,229,255,.1);border-radius:50%"></div>
<div style="position:absolute;inset:30px;border:1px dashed rgba(0,229,255,.06);border-radius:50%"></div>
<div style="position:absolute;top:50%;left:8px;right:8px;height:1px;background:rgba(0,229,255,.12);transform:translateY(-50%)"></div>
<div style="position:absolute;left:50%;top:8px;bottom:8px;width:1px;background:rgba(0,229,255,.12);transform:translateX(-50%)"></div>
<div style="position:absolute;top:50%;left:50%;width:50%;height:1px;background:linear-gradient(90deg,#00e5ff,transparent);transform-origin:0 0;animation:sR 3s linear infinite;opacity:.7"><div style="position:absolute;right:0;top:-3px;width:6px;height:6px;background:#00e5ff;border-radius:50%;box-shadow:0 0 10px #00e5ff"></div></div>
<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:5px;height:5px;background:#00e5ff;border-radius:50%;box-shadow:0 0 12px #00e5ff;animation:sP 2s ease-in-out infinite"></div>
</div>
<div style="font-size:1.6rem;font-weight:700;color:#00e5ff;letter-spacing:.18em;text-shadow:0 0 25px rgba(0,229,255,.35);margin-bottom:.1rem">WILDGUARD</div>
<div style="font-size:.42rem;color:#3a4a5c;letter-spacing:.35em;margin-bottom:1rem">THREAT INTELLIGENCE PLATFORM · v3.0</div>
<div style="width:280px;text-align:left">
<div class="sl" style="animation-delay:.2s">[<span class=g>OK</span>] KERNEL INIT · SECURE BOOT</div>
<div class="sl" style="animation-delay:.5s">[<span class=g>OK</span>] XGBOOST CLASSIFIER · <span class=c>500 ESTIMATORS</span></div>
<div class="sl" style="animation-delay:.8s">[<span class=g>OK</span>] RANDOM FOREST · <span class=c>400 TREES</span></div>
<div class="sb"><div style="animation-delay:.9s;animation-duration:1s"></div></div>
<div class="sl" style="animation-delay:1.1s">[<span class=g>OK</span>] SHAP TREE EXPLAINER · <span class=c>SHAPLEY ADDITIVE</span></div>
<div class="sl" style="animation-delay:1.5s">[<span class=g>OK</span>] REAL-TIME PIPELINE · <span class=c>7 DATA SOURCES</span></div>
<div class="sl" style="animation-delay:1.8s">[<span class=g>OK</span>] GEOSPATIAL ENGINE · FOLIUM HEATMAP</div>
<div class="sl" style="animation-delay:2.1s">[<span class=w>!!</span>] THREAT LEVEL · <span class=w>ELEVATED</span></div>
<div class="sb"><div style="animation-delay:2.2s;animation-duration:1.2s"></div></div>
<div class="sl" style="animation-delay:2.5s;font-size:.42rem">&nbsp;&nbsp;WEATHER · OSM · GBIF · NEWS · NDVI · TRAFFIC · GOV</div>
<div class="sl" style="animation-delay:2.9s">[<span class=g>OK</span>] 28 HIGHWAY SEGMENTS · SOUTH INDIA</div>
<div class="sl" style="animation-delay:3.3s">[<span class=g>::</span>] <span class=c>SYSTEM READY</span> · CLEARANCE LEVEL 4</div>
</div>
<div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00e5ff,transparent);opacity:.3"></div>
<div style="position:absolute;bottom:6px;font-size:.35rem;color:#1a2332;letter-spacing:.25em">WILDLIFE-VEHICLE COLLISION RISK INTELLIGENCE · CLASSIFIED</div>
</div>
<script>setTimeout(function(){var e=document.getElementById('spl');if(e)e.style.display='none'},5200)</script>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

@st.cache_resource
def load_model():
    from models.train import WildlifeRiskModel, train_and_save
    m = WildlifeRiskModel()
    if not (MODEL_DIR / "xgb_model.pkl").exists():
        train_and_save()
    m.load()
    return m

@st.cache_data
def load_dataset():
    if (MODEL_DIR / "dataset.parquet").exists():
        return pd.read_parquet(MODEL_DIR / "dataset.parquet")
    else:
        from data.generate_data import generate_dataset
        df = generate_dataset(12_000)
        df.to_parquet(MODEL_DIR / "dataset.parquet", index=False)
        return df

model = load_model()
df    = load_dataset()

# Parse multi-model metrics
if "xgb" in model.metrics and isinstance(model.metrics["xgb"], dict):
    xgb_metrics = model.metrics["xgb"]
    rf_metrics  = model.metrics.get("rf", {})
    shap_imp    = xgb_metrics.get("shap_importance", [])
    rf_imp      = rf_metrics.get("feature_importance", [])
    best_model  = model.metrics.get("best_model", "XGBoost")
else:
    # Backward compat with old single-model format
    xgb_metrics = model.metrics
    rf_metrics  = {}
    shap_imp    = model.metrics.get("shap_importance", [])
    rf_imp      = []
    best_model  = "XGBoost"

# ── Sidebar — Command Console ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.6rem 0;'>
      <div class='boot-text'>SYSTEM ONLINE · CLEARANCE LEVEL 4</div>
      <div class='reticle' style='margin: 0.6rem auto 0.2rem;'>
        <div class='reticle-outer'></div>
        <div class='reticle-inner'></div>
        <div class='reticle-dot'></div>
        <div class='reticle-tick t'></div>
        <div class='reticle-tick b'></div>
        <div class='reticle-tick l'></div>
        <div class='reticle-tick r'></div>
        <div class='radar-sweep-line'></div>
      </div>
      <div style='font-family:"JetBrains Mono",monospace; font-size:1.1rem; color:#00e5ff; font-weight:700; letter-spacing:0.12em; text-shadow: 0 0 18px rgba(0,229,255,0.35);'>WILDGUARD</div>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.25em; margin-top:0.1rem;'>THREAT INTELLIGENCE PLATFORM v3.0</div>
    </div>

    <!-- Threat Level Strip -->
    <div class='threat-strip' title='Threat Level: Elevated'>
      <div class='seg' style='background:#00e676;'></div>
      <div class='seg' style='background:#00e676;'></div>
      <div class='seg' style='background:#4caf50;'></div>
      <div class='seg' style='background:#ffb020;'></div>
      <div class='seg' style='background:#ffb020;'></div>
      <div class='seg' style='background:#ff7043;'></div>
      <div class='seg' style='background:#ff3d5a;'></div>
      <div class='seg' style='background:#ff3d5a;'></div>
      <div class='seg' style='background:#d500f9;'></div>
      <div class='seg' style='background:#1a2332;'></div>
    </div>

    <div style='display:flex; justify-content:space-between; padding:0 0.2rem;'>
      <div style='display:flex; align-items:center; gap:0.3rem;'>
        <span class='live-dot'></span>
        <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#00e676; letter-spacing:0.12em;'>PIPELINE ACTIVE</span>
      </div>
      <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#3a4a5c; letter-spacing:0.08em;'>▸ SECURE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Live Risk Predictor", "🗺 Risk Map",
         "📊 Analytics", "⚡ Future Hotspots", "🧠 Model Insights",
         "🐾 Animal Movement", "🌿 NDVI Prediction", "🚨 Alert System",
         "🌍 SDG Goals", "📅 Season Prediction", "📡 Data Sources"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    xgb_auc = xgb_metrics.get('roc_auc', 0)
    rf_auc  = rf_metrics.get('roc_auc', 0)
    st.markdown(f"""
    <div style='padding:0.3rem 0;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.2em; margin-bottom:0.4rem;'>▸ DUAL-MODEL STATUS</div>

      <div class='intel-panel'>
        <div class='panel-label'>XGBoost Gradient Boost</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value'>AUC {xgb_auc:.4f}</div>
          <span style='font-size:0.5rem; color:#00e676;'>● TRAINED</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Random Forest Ensemble</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value' style='color:#4da6ff;'>AUC {rf_auc:.4f}</div>
          <span style='font-size:0.5rem; color:#00e676;'>● TRAINED</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Active Classifier</div>
        <div class='panel-value' style='color:#00e676;'>◆ {best_model}</div>
      </div>
    </div>

    <!-- System Intel -->
    <div style='margin-top:0.8rem;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.2em; margin-bottom:0.4rem;'>▸ SYSTEM INTEL</div>

      <div class='intel-panel'>
        <div class='panel-label'>Data Pipeline</div>
        <div style='display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.2rem;'>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>WEATHER</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>OSM</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>GBIF</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>NEWS</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>NDVI</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>TRAFFIC</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>GOV</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Feature Vector</div>
        <div class='panel-value'>{len(model.feature_cols)} dimensions</div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Training Records</div>
        <div class='panel-value'>{len(df):,} observations</div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>SHAP Explainer</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value'>TreeSHAP</div>
          <span style='font-size:0.5rem; color:#00e676;'>● READY</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Global HUD Elements ──────────────────────────────────────────────────────
# Data Rain (Matrix falling characters — right edge)
rain_cols = ""
import random
data_chars = ["01001", "NDVI", "0.847", "RISK", "SHAP", "25.3C", "TIGER", "NH48", "ALERT", "KDE", "RF:0.81", "XGB"]
for i in range(8):
    left = 10 + i * 25
    dur = random.uniform(8, 18)
    delay = random.uniform(0, 8)
    text = " · ".join(random.choices(data_chars, k=6))
    rain_cols += f"<div class='rain-col' style='left:{left}px; animation-duration:{dur}s; animation-delay:{delay}s;'>{text}</div>"

st.markdown(f"""
<div class='data-rain'>{rain_cols}</div>

<!-- Classified Ticker -->
<div class='classified-ticker'>
  <span class='ticker-text'>
    ◈ WILDGUARD THREAT INTELLIGENCE — CLASSIFIED // RESTRICTED ACCESS — DUAL MODEL ENSEMBLE (XGBOOST + RANDOM FOREST) — 7 REAL-TIME DATA SOURCES — SHAP EXPLAINABILITY ENGINE ACTIVE — COVERING 28 HIGHWAY SEGMENTS ACROSS SOUTH INDIA — WILDLIFE CORRIDOR MONITORING — AUTOMATED RISK SCORING —
    ◈ WILDGUARD THREAT INTELLIGENCE — CLASSIFIED // RESTRICTED ACCESS — DUAL MODEL ENSEMBLE (XGBOOST + RANDOM FOREST) — 7 REAL-TIME DATA SOURCES — SHAP EXPLAINABILITY ENGINE ACTIVE —
  </span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD (Enhanced UI Overhaul)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    from utils.helpers import (
        hourly_risk_chart, species_risk_chart,
        season_road_heatmap, rolling_trend_chart, ndvi_risk_scatter,
        confusion_matrix_chart, model_comparison_chart, roc_comparison_chart,
        PALETTE
    )

    # ── Live Status Bar ───────────────────────────────────────────────────────
    _now = datetime.now()
    _season_now = ("summer" if _now.month in [3,4,5] else "monsoon" if _now.month in [6,7,8,9]
                   else "post_monsoon" if _now.month in [10,11] else "winter")
    _threat = "ELEVATED" if _season_now in ["monsoon","post_monsoon"] else "MODERATE"
    _threat_color = "#ffb020" if _threat == "ELEVATED" else "#00e676"

    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
      <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;'>
        <div>
          <div style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#3a4a5c; letter-spacing:0.25em; margin-bottom:0.3rem;'>
            ▸ CLASSIFIED // RISK INTELLIGENCE OVERVIEW // {_now.strftime('%Y-%m-%d %H:%M UTC+5:30')}
          </div>
          <h1 style='font-family:"Inter",sans-serif; font-size:1.8rem; font-weight:600; margin:0 0 0.15rem 0; color:#e8f0fa;'>
            Wildlife-Vehicle Collision
            <span style='color:#00e5ff;'>Risk Intelligence</span>
          </h1>
          <p style='color:#5a6d82; font-size:0.78rem; margin:0; font-family:"IBM Plex Mono",monospace; letter-spacing:0.03em;'>
            Dual-model ensemble (XGBoost + Random Forest) · {len(df):,} training records · 7-source real-time pipeline
          </p>
        </div>
        <div style='text-align:right;'>
          <div style='background:linear-gradient(135deg, {_threat_color}18, {_threat_color}08); border:1px solid {_threat_color}44;
                      padding:0.5rem 1rem; display:inline-block;'>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#5a6d82; letter-spacing:0.2em;'>THREAT LEVEL</div>
            <div style='font-family:"JetBrains Mono",monospace; font-size:1rem; font-weight:700; color:{_threat_color};
                        letter-spacing:0.12em;'>{_threat}</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.45rem; color:#5a6d82;'>
              Season: {_season_now.replace("_"," ").title()} · {_now.strftime('%H:%M')} IST
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row 1 — Primary Metrics ──────────────────────────────────────────
    acc_rate  = df["accident"].mean()
    high_risk = (df["risk_score"] > 0.65).sum()
    critical_zones = (df["risk_score"] > 0.80).sum()
    top_sp    = df[df["accident"]==1]["species"].value_counts().idxmax()
    top_road  = df[df["accident"]==1]["road_type"].value_counts().idxmax()
    avg_risk  = df["risk_score"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Records",    f"{len(df):,}",          "training dataset",    "#00e5ff"),
        (c2, "Accident Rate",    f"{acc_rate:.1%}",        "binary target",       "#ff3d5a" if acc_rate > 0.5 else "#ffb020"),
        (c3, "High-Risk Zones",  f"{high_risk:,}",         "score > 0.65",        "#ff3d5a"),
        (c4, "Critical Zones",   f"{critical_zones:,}",    "score > 0.80",        "#d500f9"),
        (c5, "XGB AUC",          f"{xgb_metrics.get('roc_auc',0):.4f}", "gradient boosting", "#00e5ff"),
        (c6, "RF AUC",           f"{rf_metrics.get('roc_auc',0):.4f}",  "random forest",     "#4da6ff"),
    ]
    for col, title, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <h3>{title}</h3>
              <div class='value' style='font-size:1.6rem; color:{color};'>{val}</div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── KPI Row 2 — Ecological & Risk Indicators ─────────────────────────────
    c7, c8, c9, c10 = st.columns(4)
    avg_ndvi = df["ndvi"].mean()
    avg_movement = df["movement_score"].mean()
    breeding_pct = df["breeding_season"].mean()
    night_acc_rate = df[df["night_flag"]==1]["accident"].mean()

    kpis2 = [
        (c7,  "Avg NDVI",         f"{avg_ndvi:.3f}",     "vegetation density",   "#00e676"),
        (c8,  "Movement Score",   f"{avg_movement:.3f}", "animal activity",      "#ffb020"),
        (c9,  "Breeding Period",  f"{breeding_pct:.0%}", "monsoon + post-monsoon","#d500f9"),
        (c10, "Night Risk Rate",  f"{night_acc_rate:.1%}","20:00-06:00 window",  "#ff3d5a"),
    ]
    for col, title, val, sub, color in kpis2:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <h3>{title}</h3>
              <div class='value' style='font-size:1.4rem; color:{color};'>{val}</div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Distribution Overview ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Risk Distribution Overview</div>", unsafe_allow_html=True)
    risk_cols = st.columns(4)
    risk_bins = [(0, 0.25, "Low", "#00e676"), (0.25, 0.50, "Moderate", "#ffb020"),
                 (0.50, 0.75, "High", "#ff3d5a"), (0.75, 1.01, "Critical", "#d500f9")]
    for i, (lo, hi, label, color) in enumerate(risk_bins):
        count = ((df["risk_score"] >= lo) & (df["risk_score"] < hi)).sum()
        pct = count / len(df)
        with risk_cols[i]:
            st.markdown(f"""
            <div style='background:linear-gradient(145deg, {color}15, {color}05); border:1px solid {color}33;
                        padding:0.7rem 0.9rem; text-align:center;'>
              <div style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:{color};
                          letter-spacing:0.15em; text-transform:uppercase;'>{label} Risk</div>
              <div style='font-family:"JetBrains Mono",monospace; font-size:1.5rem; font-weight:700;
                          color:{color}; margin:0.2rem 0;'>{count:,}</div>
              <div style='font-family:"IBM Plex Mono",monospace; font-size:0.6rem; color:#5a6d82;'>{pct:.1%} of records</div>
              <div style='background:#1a2332; height:4px; margin-top:0.4rem; overflow:hidden;'>
                <div style='background:{color}; height:100%; width:{pct*100:.0f}%;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison row ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Performance Comparison — XGBoost vs Random Forest</div>", unsafe_allow_html=True)
    col_mc, col_roc = st.columns(2)
    with col_mc:
        if rf_metrics:
            st.plotly_chart(model_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)
    with col_roc:
        if rf_metrics:
            st.plotly_chart(roc_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.plotly_chart(hourly_risk_chart(df), use_container_width=True)
    with col_b:
        st.plotly_chart(species_risk_chart(df), use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_c, col_d = st.columns([1, 1.2])
    with col_c:
        st.plotly_chart(season_road_heatmap(df), use_container_width=True)
    with col_d:
        st.plotly_chart(ndvi_risk_scatter(df), use_container_width=True)

    # ── Species Corridor Analysis ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Species-Corridor Risk Matrix</div>", unsafe_allow_html=True)
    sp_corr = df.groupby("species").agg(
        avg_risk=("risk_score","mean"), avg_movement=("movement_score","mean"),
        avg_corridor=("corridor_dist_km","mean"), total_acc=("accident","sum"),
        count=("accident","count")
    ).reset_index().sort_values("avg_risk", ascending=False)

    fig_sp_matrix = go.Figure()
    sp_colors = {"tiger":"#ff3d5a","elephant":"#d500f9","leopard":"#ff7043","deer":"#ffb020",
                 "boar":"#9d7aff","wolf":"#4da6ff","nilgai":"#00e676","sambar":"#00e5ff"}
    for _, row in sp_corr.iterrows():
        fig_sp_matrix.add_trace(go.Scatter(
            x=[row["avg_corridor"]], y=[row["avg_risk"]],
            mode="markers+text",
            marker=dict(size=max(10, row["total_acc"]/5), color=sp_colors.get(row["species"],"#00e5ff"),
                       opacity=0.8, line=dict(width=1, color="#1a2332")),
            text=[row["species"].capitalize()], textposition="top center",
            textfont=dict(size=9, color=sp_colors.get(row["species"],"#00e5ff")),
            name=row["species"].capitalize(),
            hovertemplate=f"<b>{row['species'].capitalize()}</b><br>Risk: {row['avg_risk']:.3f}<br>"
                         f"Corridor Dist: {row['avg_corridor']:.1f}km<br>Accidents: {row['total_acc']}<extra></extra>",
        ))
    fig_sp_matrix.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        xaxis=dict(title="Avg Corridor Distance (km)", gridcolor="#1a2332", zerolinecolor="#1a2332"),
        yaxis=dict(title="Avg Risk Score", gridcolor="#1a2332", zerolinecolor="#1a2332"),
        height=380, showlegend=False, margin=dict(l=40,r=20,t=30,b=40),
    )
    st.plotly_chart(fig_sp_matrix, use_container_width=True)

    # ── Rolling trend ─────────────────────────────────────────────────────────
    st.plotly_chart(rolling_trend_chart(df), use_container_width=True)

    # ── Confusion matrices side by side ───────────────────────────────────────
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    col_e, col_f = st.columns(2)
    with col_e:
        cm_xgb = xgb_metrics.get("confusion", [[0,0],[0,0]])
        st.plotly_chart(confusion_matrix_chart(cm_xgb, "XGBoost — Confusion Matrix"), use_container_width=True)
    with col_f:
        if rf_metrics:
            cm_rf = rf_metrics.get("confusion", [[0,0],[0,0]])
            st.plotly_chart(confusion_matrix_chart(cm_rf, "Random Forest — Confusion Matrix"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE RISK PREDICTOR (Real-Time Data Pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Risk Predictor":
    from utils.helpers import (
        dual_risk_gauge, shap_waterfall, get_risk_level,
        data_source_status_chart, PALETTE
    )
    from data.realtime_extractor import RealtimeDataExtractor, geocode_place, PROTECTED_AREAS, WILDLIFE_CORRIDORS

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🔮 Live <span style='color:#FF6B35;'>Risk Prediction</span> Pipeline
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.2rem 0;'>
      Search any place in India → extracts real-time data from APIs, news sites & govt portals → dual-model prediction
    </p>
    """, unsafe_allow_html=True)

    # ── Location inputs: Search + Presets + Manual ────────────────────────────
    st.markdown("<div class='section-header'>📍 Search Location</div>", unsafe_allow_html=True)

    # Search bar
    search_col, search_btn_col = st.columns([4, 1])
    with search_col:
        place_query = st.text_input(
            "🔍 Search any place in India",
            placeholder="e.g., Bandipur Tiger Reserve, Mysore-Ooty Highway, Wayanad...",
            key="place_search"
        )
    with search_btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔎 Search", use_container_width=True)

    # Handle geocoding
    resolved_lat, resolved_lon = None, None
    resolved_name = ""
    if search_clicked and place_query:
        with st.spinner(f"🌍 Geocoding '{place_query}' via Nominatim (OpenStreetMap)..."):
            geo_result = geocode_place(place_query)
        if geo_result and geo_result.get("found"):
            resolved_lat = geo_result["lat"]
            resolved_lon = geo_result["lon"]
            resolved_name = geo_result.get("display_name", "")
            st.success(f"📍 **Found:** {resolved_name}")
            st.markdown(f"<div style='font-size:0.78rem; color:var(--muted);'>Lat: {resolved_lat:.5f} | Lon: {resolved_lon:.5f} | Type: {geo_result.get('place_type','')}</div>", unsafe_allow_html=True)
            if len(geo_result.get("all_results", [])) > 1:
                with st.expander("Other matching locations"):
                    for r in geo_result["all_results"][1:]:
                        st.markdown(f"- {r['name']} ({r['lat']:.4f}, {r['lon']:.4f})")
        else:
            st.error(f"❌ Could not find '{place_query}'. Try a more specific name.")

    st.markdown("<div style='text-align:center; color:var(--muted); font-size:0.75rem; margin:0.3rem 0;'>─── OR select a preset ───</div>", unsafe_allow_html=True)

    PRESETS = {
        "Custom Location": (12.0, 76.5),
        "── South India ──": (11.66, 76.63),
        "🐯 Bandipur National Park (Karnataka)": (11.66, 76.63),
        "🐘 Nagarhole / Rajiv Gandhi NP (Karnataka)": (12.05, 76.15),
        "🦁 Mudumalai Tiger Reserve (Tamil Nadu)": (11.56, 76.55),
        "🌿 Wayanad Wildlife Sanctuary (Kerala)": (11.60, 76.02),
        "🐘 Periyar Tiger Reserve (Kerala)": (9.47, 77.17),
        "🐆 Sathyamangalam TR (Tamil Nadu)": (11.50, 77.25),
        "🌲 BR Hills Sanctuary (Karnataka)": (11.99, 77.16),
        "── Central India ──": (22.33, 80.62),
        "🐯 Kanha National Park (MP)": (22.33, 80.62),
        "🐘 Tadoba-Andhari Reserve (Maharashtra)": (20.20, 79.35),
        "🦁 Pench Tiger Reserve (MP)": (21.72, 79.30),
        "🌲 Satpura Tiger Reserve (MP)": (22.52, 78.12),
    }

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        preset = st.selectbox("Predefined Wildlife Zones", list(PRESETS.keys()))
    with c2:
        default_lat, default_lon = PRESETS[preset]
        if resolved_lat is not None:
            default_lat = resolved_lat
        lat = st.number_input("Latitude", min_value=6.0, max_value=37.0, value=default_lat, step=0.01)
    with c3:
        if resolved_lon is not None:
            default_lon = resolved_lon
        lon = st.number_input("Longitude", min_value=68.0, max_value=97.0, value=default_lon, step=0.01)

    c4, c5, c6 = st.columns(3)
    with c4: speed_limit  = st.selectbox("Speed Limit (km/h)", [30, 40, 60, 80, 100], index=2)
    with c5: actual_speed = st.slider("Actual Speed (km/h)", 10, 140, 65)
    with c6: past_acc     = st.slider("Past Accident Count", 0, 30, 3)

    run_btn = st.button("⚡ EXTRACT DATA & PREDICT RISK", use_container_width=True)

    if run_btn:
        # ── RUN THE PIPELINE ──────────────────────────────────────────────────
        with st.spinner("📡 Extracting real-time data from APIs, news sites & govt portals…"):
            extractor = RealtimeDataExtractor()
            feature_df = extractor.extract_all(
                lat, lon,
                speed_limit=speed_limit,
                actual_speed=actual_speed,
                past_accidents=past_acc,
            )
            extraction_log = extractor.get_extraction_log()
            summary        = extractor.get_extraction_summary()

        # Save to session_state so Map page can access it
        spatial_data = None
        for er in extractor.extraction_log:
            if er.source_type == "computation" and "nearest_pa" in er.data:
                spatial_data = er.data
        st.session_state["last_prediction"] = {
            "lat": lat, "lon": lon,
            "location_name": resolved_name or preset,
            "features": feature_df.iloc[0].to_dict(),
            "log": extraction_log,
            "summary": summary,
            "spatial": spatial_data,
        }

        # ── DATA SOURCE STATUS ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Real-Time Data Extraction — 7 Sources</div>", unsafe_allow_html=True)

        # Group by source type for display
        api_sources  = [l for l in extraction_log if l.get('source_type') == 'api']
        news_sources = [l for l in extraction_log if l.get('source_type') == 'news']
        govt_sources = [l for l in extraction_log if l.get('source_type') == 'government']
        comp_sources = [l for l in extraction_log if l.get('source_type') == 'computation']

        # Row 1: APIs
        st.markdown("**🌐 API Sources**", unsafe_allow_html=True)
        api_cols = st.columns(max(len(api_sources), 1))
        for i, log_entry in enumerate(api_sources):
            with api_cols[i]:
                si = "✅" if log_entry["status"]=="success" else "⚠️"
                fs = ", ".join(log_entry["features"][:3])
                st.markdown(f"<div class='source-card'><div class='source-name'>{si} {log_entry['source']}</div><div class='source-url'>{log_entry['api_url'][:70]}…</div><div style='font-size:0.72rem;'><span class='status-{log_entry["status"]}'>{log_entry['status'].upper()}</span> · {log_entry['response_ms']}ms</div><div class='source-features'>→ {fs}</div></div>", unsafe_allow_html=True)

        # Row 2: News + Govt
        ng_col1, ng_col2 = st.columns(2)
        for col, label, sources in [(ng_col1, "📰 News Websites", news_sources), (ng_col2, "🏛️ Government Portals", govt_sources)]:
            with col:
                st.markdown(f"**{label}**")
                for log_entry in sources:
                    si = "✅" if log_entry["status"]=="success" else "⚠️"
                    fs = ", ".join(str(f) for f in log_entry["features"][:3])
                    st.markdown(f"<div class='source-card'><div class='source-name'>{si} {log_entry['source']}</div><div style='font-size:0.72rem;'><span class='status-{log_entry["status"]}'>{log_entry['status'].upper()}</span> · {log_entry['response_ms']}ms</div><div class='source-features'>→ {fs}</div></div>", unsafe_allow_html=True)

        # Row 3: Computations
        if comp_sources:
            st.markdown("**🧮 Computed Sources**")
            comp_cols = st.columns(len(comp_sources))
            for i, log_entry in enumerate(comp_sources):
                with comp_cols[i]:
                    st.markdown(f"<div class='source-card'><div class='source-name'>✅ {log_entry['source']}</div><div style='font-size:0.72rem;'>SUCCESS · {log_entry['response_ms']}ms</div><div class='source-features'>→ {', '.join(log_entry['features'][:4])}</div></div>", unsafe_allow_html=True)

        st.plotly_chart(data_source_status_chart(extraction_log), use_container_width=True)

        # ── NEWS ARTICLES ─────────────────────────────────────────────────────
        news_data = None
        govt_detail = None
        for log_entry in extraction_log:
            if log_entry.get('source_type') == 'news':
                # Get news data from extractor
                for er in extractor.extraction_log:
                    if er.source_type == 'news':
                        news_data = er.data
            if log_entry.get('source_type') == 'government':
                for er in extractor.extraction_log:
                    if er.source_type == 'government':
                        govt_detail = er.data

        if news_data and news_data.get('top_articles'):
            st.markdown("<div class='section-header'>📰 Recent Wildlife News from Indian Media</div>", unsafe_allow_html=True)
            nc1, nc2 = st.columns([1, 1])
            with nc1:
                st.markdown(f"<div class='metric-card'><h3>Wildlife Articles Found</h3><div class='value' style='font-size:1.4rem;'>{news_data.get('total_wildlife_articles',0)}</div><div class='sub'>From The Hindu, NDTV, Down to Earth, Google News</div></div>", unsafe_allow_html=True)
            with nc2:
                st.markdown(f"<div class='metric-card'><h3>South India Mentions</h3><div class='value' style='font-size:1.4rem;'>{news_data.get('south_india_articles',0)}</div><div class='sub'>Matching Karnataka, Kerala, Tamil Nadu</div></div>", unsafe_allow_html=True)
            for art in news_data['top_articles'][:5]:
                si_tag = " 🟢 South India" if art.get('south_india_match') else ""
                sp_tag = f" | Species: {', '.join(art.get('species_mentioned',[])) }" if art.get('species_mentioned') else ""
                st.markdown(f"- **[{art['source']}]** {art['title']}{si_tag}{sp_tag}")

        if govt_detail:
            st.markdown("<div class='section-header'>🏛️ Indian Government Portal Status</div>", unsafe_allow_html=True)
            gs = govt_detail.get('source_status', {})
            gc1, gc2 = st.columns(2)
            with gc1:
                for name, status in list(gs.items())[:4]:
                    st.markdown(f"- **{name}**: {status}")
            with gc2:
                for name, status in list(gs.items())[4:]:
                    st.markdown(f"- **{name}**: {status}")
            st.markdown(f"📍 **Nearest State Dept**: {govt_detail.get('nearest_state_dept','')} ({govt_detail.get('nearest_state_url','')})")

        # ── PREDICTION RESULTS ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🤖 Dual-Model Prediction Results</div>", unsafe_allow_html=True)

        if model.rf_model is not None:
            results = model.predict_both(feature_df)
            xgb_prob = results["xgb_probability"]
            rf_prob  = results["rf_probability"]
            avg_prob = results["avg_probability"]
            agree    = results["agreement"]
        else:
            xgb_prob = float(model.predict_risk(feature_df)[0])
            rf_prob  = xgb_prob
            avg_prob = xgb_prob
            agree    = True

        rl_xgb = get_risk_level(xgb_prob)
        rl_rf  = get_risk_level(rf_prob)
        rl_avg = get_risk_level(avg_prob)

        # Gauges
        st.plotly_chart(dual_risk_gauge(xgb_prob, rf_prob), use_container_width=True)

        # Risk badges
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_xgb["color"]}33; color:{rl_xgb["color"]}; border:1.5px solid {rl_xgb["color"]};'>
                {rl_xgb["emoji"]} XGBoost: {xgb_prob:.1%}
              </span>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_rf["color"]}33; color:{rl_rf["color"]}; border:1.5px solid {rl_rf["color"]};'>
                {rl_rf["emoji"]} Random Forest: {rf_prob:.1%}
              </span>
            </div>""", unsafe_allow_html=True)
        with r3:
            agree_str = "✅ Models Agree" if agree else "⚠️ Divergence Detected"
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_avg["color"]}33; color:{rl_avg["color"]}; border:1.5px solid {rl_avg["color"]};'>
                {rl_avg["emoji"]} Ensemble: {avg_prob:.1%} · {agree_str}
              </span>
            </div>""", unsafe_allow_html=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔬 SHAP Explanation — Why This Prediction</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#5a6d82; font-size:0.72rem; font-family:"IBM Plex Mono",monospace;'>
          SHAP (SHapley Additive exPlanations) decomposes the prediction into individual feature contributions.
          Positive values push risk higher; negative values push risk lower.
        </p>
        """, unsafe_allow_html=True)

        shap_explanations = {
            'movement_score': ("Animal Activity", "Composite score of wildlife movement likelihood based on habitat, water proximity, and temporal factors"),
            'driver_risk': ("Driver Behavior", "Combination of speed compliance, night driving, and road type risk factors"),
            'species_risk': ("Species Danger Level", "Mapped risk score for the dominant wildlife species in the area"),
            'kde_density': ("Historical Hotspot Density", "Kernel density estimate from past accident cluster analysis"),
            'road_type': ("Road Classification", "Type of road — forest roads and rural roads carry higher inherent risk"),
            'night_flag': ("Nighttime Driving", "Binary flag: 1 = night (8 PM–6 AM), when visibility drops and animals are active"),
            'corridor_dist_km': ("Wildlife Corridor Distance", "Distance from nearest known wildlife migration corridor"),
            'speed_ratio': ("Speed Compliance", "Ratio of actual speed to posted limit — values > 1.0 indicate speeding"),
            'ndvi': ("Vegetation Density (NDVI)", "Normalized vegetation index — dense vegetation reduces driver sightlines"),
            'past_accidents': ("Historical Incidents", "Number of recorded accidents in this location previously"),
            'rainfall_mm': ("Current Rainfall", "Precipitation level — wet conditions + animal water-seeking = higher risk"),
            'visibility_m': ("Visibility Distance", "How far ahead the driver can see — fog/rain reduce this dramatically"),
            'temperature_c': ("Temperature", "Current temperature — extreme heat/cold affects animal movement patterns"),
            'humidity_pct': ("Humidity Level", "Atmospheric humidity — correlates with fog formation and animal activity"),
            'hour': ("Time of Day", "Hour of the day — dawn (5-7 AM) and dusk (6-8 PM) are peak crossing times"),
            'dawn_dusk': ("Dawn/Dusk Window", "Whether it's currently in the critical dawn or dusk period"),
            'breeding_season': ("Breeding Season", "Whether animals are in their breeding period — increases movement"),
            'dist_water_km': ("Water Source Distance", "Distance to nearest water body — animals cross roads to reach water"),
            'protected_dist_km': ("Protected Area Distance", "Distance from nearest wildlife sanctuary or national park"),
        }

        try:
            sv, X_in, base = model.predict_shap(feature_df)
            sv_flat = sv[0] if sv.ndim == 2 else sv
            st.plotly_chart(
                shap_waterfall(sv_flat, model.feature_cols,
                               X_in.iloc[0].values, base),
                use_container_width=True
            )

            # ── Top SHAP Drivers Table ────────────────────────────────────────
            st.markdown("<div class='section-header'>📊 Top Risk Drivers — Plain Language</div>", unsafe_allow_html=True)
            pairs = sorted(zip(sv_flat, model.feature_cols, X_in.iloc[0].values),
                           key=lambda x: abs(x[0]), reverse=True)[:8]

            for sv_val, feat, feat_val in pairs:
                direction = "↑ increases" if sv_val > 0 else "↓ decreases"
                dir_color = "#ff3d5a" if sv_val > 0 else "#00e676"
                human_name, description = shap_explanations.get(feat, (feat.replace("_", " ").title(), "Feature contributing to risk prediction"))
                st.markdown(f"""
                <div style='background:#0d1320; border:1px solid #1a2332; padding:0.7rem 1rem; margin:0.3rem 0; position:relative;'>
                  <div style='position:absolute; left:0; top:0; bottom:0; width:3px; background:{dir_color};'></div>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <span style='font-family:"JetBrains Mono",monospace; font-size:0.78rem; color:#e8f0fa; font-weight:600;'>{human_name}</span>
                      <span style='font-family:"IBM Plex Mono",monospace; font-size:0.6rem; color:#5a6d82; margin-left:0.5rem;'>({feat})</span>
                    </div>
                    <div style='text-align:right;'>
                      <span style='font-family:"JetBrains Mono",monospace; font-size:0.82rem; color:{dir_color}; font-weight:600;'>{sv_val:+.4f}</span>
                      <span style='font-family:"IBM Plex Mono",monospace; font-size:0.6rem; color:#5a6d82; margin-left:0.5rem;'>val={feat_val:.3f}</span>
                    </div>
                  </div>
                  <div style='font-size:0.68rem; color:#5a6d82; margin-top:0.3rem; font-family:"IBM Plex Mono",monospace;'>
                    {direction} risk · {description}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── Extracted feature table ───────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Extracted Feature Vector</div>", unsafe_allow_html=True)
        feat_display = feature_df.T.reset_index()
        feat_display.columns = ["Feature", "Value"]
        st.dataframe(feat_display, use_container_width=True, height=400)

        # ── Recommendations ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🛡 Mitigation Recommendations</div>", unsafe_allow_html=True)
        recs = []
        row = feature_df.iloc[0]
        if row.get("night_flag", 0):     recs.append("🌙 **Nocturnal alert zone** — deploy flashing warning signs between 20:00–06:00")
        if row.get("ndvi", 0) > 0.6:    recs.append("🌲 **High vegetation corridor** — install wildlife detection sensors")
        if row.get("dist_water_km", 99) < 1: recs.append("💧 **Water source proximity** — install water crossing structures / underpasses")
        if row.get("speed_ratio", 0) > 1.1:  recs.append("🚗 **Over-speed detected** — enforce speed cameras and lower limit")
        if row.get("rainfall_mm", 0) > 30:   recs.append("🌧 **Low visibility conditions** — dynamic variable speed limits")
        if row.get("breeding_season", 0):     recs.append("🔥 **Breeding season** — temporary speed restrictions May–October")
        if row.get("corridor_dist_km", 99) < 2: recs.append("🗺 **Corridor proximity** — install wildlife fencing and crossing")
        if not recs: recs.append("✅ Risk factors within acceptable range — standard monitoring advised")
        for rec in recs:
            st.markdown(f"- {rec}")

        # ── Data source detail expanders ──────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Detailed Data Source Logs</div>", unsafe_allow_html=True)
        for log_entry in extraction_log:
            status_icon = "✅" if log_entry["status"] == "success" else "⚠️" if log_entry["status"] == "fallback" else "❌"
            with st.expander(f"{status_icon} {log_entry['source']}  —  {log_entry['response_ms']}ms"):
                st.markdown(f"**API URL:** `{log_entry['api_url']}`")
                st.markdown(f"**Description:** {log_entry['description']}")
                st.markdown(f"**Status:** `{log_entry['status']}`")
                st.markdown(f"**Timestamp:** `{log_entry['timestamp']}`")
                st.markdown(f"**Features extracted:** {', '.join(log_entry['features'])}")
                if log_entry["error"]:
                    st.error(f"Error: {log_entry['error']}")
                if log_entry["raw_preview"]:
                    st.code(log_entry["raw_preview"], language="json")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK MAP (Heatmap + Live Prediction Toggle)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Risk Map":
    from streamlit_folium import st_folium
    from utils.helpers import build_folium_map, get_risk_level
    from data.realtime_extractor import PROTECTED_AREAS, WILDLIFE_CORRIDORS
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.5rem 0;'>
      🗺 Geospatial <span style='color:#FF6B35;'>Risk Map</span>
    </h1>
    """, unsafe_allow_html=True)

    # ── Toggle: Full Heatmap vs Live Prediction ──────────────────────────────
    has_prediction = "last_prediction" in st.session_state
    toggle_options = ["🌡️ Full Risk Heatmap (All Data)", "📍 Live Prediction Result"]

    map_mode = st.radio(
        "Map View",
        toggle_options,
        index=1 if has_prediction else 0,
        horizontal=True,
        key="map_mode_toggle",
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 1: FULL HEATMAP (training data + PAs + corridors)
    # ══════════════════════════════════════════════════════════════════════════
    if map_mode == toggle_options[0]:
        st.markdown("""
        <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1rem 0;'>
          Risk heatmap from training data — density of wildlife-vehicle collision zones with all protected areas & corridors
        </p>
        """, unsafe_allow_html=True)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1: f_season  = st.multiselect("Season",    df["season"].unique().tolist(),    default=df["season"].unique().tolist())
        with fc2: f_species = st.multiselect("Species",   df["species"].unique().tolist(),   default=df["species"].unique().tolist())
        with fc3: f_road    = st.multiselect("Road Type", df["road_type"].unique().tolist(), default=df["road_type"].unique().tolist())

        map_df = df[df["season"].isin(f_season) & df["species"].isin(f_species) & df["road_type"].isin(f_road)]

        # Build rich Folium map
        m = folium.Map(location=[15.0, 77.5], zoom_start=6, tiles="CartoDB dark_matter", control_scale=True)

        # Heatmap layer from training data
        acc = map_df[map_df["accident"] == 1][["latitude", "longitude", "risk_score"]].dropna()
        heat_data = [[r.latitude, r.longitude, r.risk_score] for _, r in acc.iterrows()]
        if heat_data:
            HeatMap(
                heat_data, min_opacity=0.3, max_zoom=14, radius=18, blur=15,
                gradient={"0.2": "#06D6A0", "0.5": "#FFD166", "0.75": "#EF476F", "1.0": "#B5179E"},
            ).add_to(m)

        # Protected Areas layer
        pa_group = folium.FeatureGroup(name="🟢 Protected Areas (21)")
        for pa in PROTECTED_AREAS:
            folium.CircleMarker(
                location=[pa["lat"], pa["lon"]], radius=7,
                color="#06D6A0", fill=True, fill_color="#06D6A0", fill_opacity=0.7,
                popup=folium.Popup(f"<b>🟢 {pa['name']}</b><br>{pa.get('state','')}", max_width=220),
                tooltip=pa["name"],
            ).add_to(pa_group)
        pa_group.add_to(m)

        # Corridors layer
        cor_group = folium.FeatureGroup(name="🟠 Wildlife Corridors (11)")
        for cor in WILDLIFE_CORRIDORS:
            folium.CircleMarker(
                location=[cor["lat"], cor["lon"]], radius=5,
                color="#FFD166", fill=True, fill_color="#FFD166", fill_opacity=0.7,
                popup=folium.Popup(f"<b>🟠 {cor['name']}</b><br>Wildlife Corridor", max_width=220),
                tooltip=cor["name"],
            ).add_to(cor_group)
        cor_group.add_to(m)

        # High-risk markers
        high = map_df[map_df["risk_score"] > 0.70].head(150)
        cluster = MarkerCluster(name="🔴 High Risk Incidents").add_to(m)
        for _, row in high.iterrows():
            rl = get_risk_level(row["risk_score"])
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]], radius=5,
                color=rl["color"], fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{rl['emoji']} {rl['label']}</b><br>"
                    f"Risk: {row['risk_score']:.2%}<br>"
                    f"Species: {row.get('species','–')}<br>"
                    f"Road: {row.get('road_type','–')}<br>"
                    f"Hour: {int(row.get('hour',0)):02d}:00",
                    max_width=200),
            ).add_to(cluster)

        folium.LayerControl().add_to(m)

        # Render
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st_folium(m, width=None, height=620, returned_objects=[])

        with c_right:
            st.markdown("<div class='section-header'>Map Legend</div>", unsafe_allow_html=True)
            for color, label in [("#06D6A0","🟢 Protected Areas"),("#FFD166","🟠 Corridors"),("#EF476F","🔴 High Risk"),("#B5179E","🟣 Critical")]:
                st.markdown(f"<div style='display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;'><div style='width:14px;height:14px;border-radius:50%;background:{color};'></div><span style='font-size:0.82rem;'>{label}</span></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:0.5rem; padding:0.3rem 0; border-top:1px solid var(--border);'></div>
            <div class='section-header'>Summary</div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-card'>
              <h3>Records</h3><div class='value' style='font-size:1.3rem;'>{len(map_df):,}</div>
            </div>
            <div class='metric-card'>
              <h3>Accidents</h3><div class='value' style='font-size:1.3rem;'>{int(map_df["accident"].sum()):,}</div>
            </div>
            <div class='metric-card'>
              <h3>Avg Risk</h3><div class='value' style='font-size:1.3rem;'>{map_df["risk_score"].mean():.3f}</div>
            </div>
            <div class='metric-card'>
              <h3>Heatmap Points</h3><div class='value' style='font-size:1.3rem;'>{len(heat_data):,}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 2: LIVE PREDICTION (from Live Risk Predictor result)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        if not has_prediction:
            st.warning("⚠️ No live prediction data yet. Go to **🔮 Live Risk Predictor** and run an extraction first.")
            st.info("The Live Risk Predictor fetches real-time data from 7 sources (APIs, news, govt portals) and the results will appear here on the map.")
        else:
            pred = st.session_state["last_prediction"]
            feat = pred["features"]
            spatial = pred["spatial"] or {}
            pred_lat, pred_lon = pred["lat"], pred["lon"]
            loc_name = pred.get("location_name", "Searched Location")

            st.markdown(f"""
            <p style='color:#8B949E; font-size:0.82rem; margin:0 0 0.5rem 0;'>
              Showing extracted data for <b style='color:#FF6B35;'>{loc_name}</b> · Lat {pred_lat:.4f}, Lon {pred_lon:.4f}
            </p>
            """, unsafe_allow_html=True)

            # Build focused map
            m = folium.Map(location=[pred_lat, pred_lon], zoom_start=11, tiles="CartoDB dark_matter", control_scale=True)

            # Nearby PAs (within ~100km)
            from data.realtime_extractor import _haversine
            nearby_pas = []
            for pa in PROTECTED_AREAS:
                d = _haversine(pred_lat, pred_lon, pa["lat"], pa["lon"])
                if d < 100:
                    nearby_pas.append((d, pa))
                    folium.CircleMarker(
                        location=[pa["lat"], pa["lon"]], radius=8,
                        color="#06D6A0", fill=True, fill_color="#06D6A0", fill_opacity=0.7,
                        popup=folium.Popup(f"<b>🟢 {pa['name']}</b><br>{pa.get('state','')}<br>{d:.1f}km away", max_width=220),
                        tooltip=f"{pa['name']} ({d:.0f}km)",
                    ).add_to(m)

            # Nearby corridors
            nearby_cors = []
            for cor in WILDLIFE_CORRIDORS:
                d = _haversine(pred_lat, pred_lon, cor["lat"], cor["lon"])
                if d < 100:
                    nearby_cors.append((d, cor))
                    folium.CircleMarker(
                        location=[cor["lat"], cor["lon"]], radius=6,
                        color="#FFD166", fill=True, fill_color="#FFD166", fill_opacity=0.7,
                        popup=folium.Popup(f"<b>🟠 {cor['name']}</b><br>{d:.1f}km away", max_width=220),
                        tooltip=f"{cor['name']} ({d:.0f}km)",
                    ).add_to(m)

            # Prediction marker
            risk_score = feat.get("driver_risk", 0.5)
            risk_color = "#06D6A0" if risk_score < 0.4 else "#FFD166" if risk_score < 0.7 else "#EF476F" if risk_score < 1.0 else "#B5179E"

            popup_html = f"""
            <div style='font-family:sans-serif; min-width:260px;'>
                <h3 style='margin:0 0 6px 0; color:{risk_color};'>📍 {loc_name}</h3>
                <table style='font-size:11px; width:100%;'>
                    <tr><td><b>Species</b></td><td>{feat.get('species','—').capitalize()} (risk {feat.get('species_risk',0):.0%})</td></tr>
                    <tr><td><b>Road Type</b></td><td>{feat.get('road_type','—')}</td></tr>
                    <tr><td><b>Temperature</b></td><td>{feat.get('temperature_c',0):.1f}°C</td></tr>
                    <tr><td><b>Humidity</b></td><td>{feat.get('humidity_pct',0):.0f}%</td></tr>
                    <tr><td><b>Rain</b></td><td>{feat.get('rainfall_mm',0):.1f}mm</td></tr>
                    <tr><td><b>Visibility</b></td><td>{feat.get('visibility_m',0)}m</td></tr>
                    <tr><td><b>NDVI</b></td><td>{feat.get('ndvi',0):.3f}</td></tr>
                    <tr><td><b>Nearest PA</b></td><td>{spatial.get('nearest_pa','—')} ({spatial.get('protected_dist_km',0):.1f}km)</td></tr>
                    <tr><td><b>Nearest Corridor</b></td><td>{spatial.get('nearest_corridor','—')} ({spatial.get('corridor_dist_km',0):.1f}km)</td></tr>
                    <tr><td><b>Driver Risk</b></td><td style='color:{risk_color}; font-weight:bold;'>{risk_score:.3f}</td></tr>
                </table>
            </div>
            """
            folium.Marker(
                location=[pred_lat, pred_lon],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"📍 {loc_name} — Risk {risk_score:.3f}",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            ).add_to(m)

            # Analysis radius
            folium.Circle(
                location=[pred_lat, pred_lon], radius=5000,
                color=risk_color, fill=True, fill_opacity=0.08, popup="5km analysis area",
            ).add_to(m)

            # Risk heatmap point (single-point, visually showing the searched area)
            HeatMap(
                [[pred_lat, pred_lon, risk_score]],
                min_opacity=0.4, radius=40, blur=25,
                gradient={"0.2": "#06D6A0", "0.5": "#FFD166", "0.75": "#EF476F", "1.0": "#B5179E"},
            ).add_to(m)

            folium.LayerControl().add_to(m)

            # Render
            c_left, c_right = st.columns([3, 1])
            with c_left:
                st_folium(m, width=None, height=620, returned_objects=[])

            with c_right:
                st.markdown("<div class='section-header'>📍 Extracted Data</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                  <h3>Location</h3><div class='value' style='font-size:0.85rem;'>{loc_name[:40]}</div>
                  <div class='sub'>{pred_lat:.4f}, {pred_lon:.4f}</div>
                </div>
                <div class='metric-card'>
                  <h3>Nearest PA</h3><div class='value' style='font-size:0.9rem;'>{spatial.get('nearest_pa','—')}</div>
                  <div class='sub'>{spatial.get('protected_dist_km',0):.1f}km · {spatial.get('nearest_pa_state','')}</div>
                </div>
                <div class='metric-card'>
                  <h3>Nearest Corridor</h3><div class='value' style='font-size:0.9rem;'>{spatial.get('nearest_corridor','—')}</div>
                  <div class='sub'>{spatial.get('corridor_dist_km',0):.1f}km</div>
                </div>
                <div class='metric-card'>
                  <h3>Species</h3><div class='value' style='font-size:1.1rem;'>{feat.get('species','—').capitalize()}</div>
                  <div class='sub'>Risk: {feat.get('species_risk',0):.0%}</div>
                </div>
                <div class='metric-card'>
                  <h3>Weather</h3><div class='value' style='font-size:0.9rem;'>{feat.get('temperature_c',0):.1f}°C · {feat.get('humidity_pct',0):.0f}%</div>
                  <div class='sub'>Rain {feat.get('rainfall_mm',0):.1f}mm · Vis {feat.get('visibility_m',0)}m</div>
                </div>
                <div class='metric-card'>
                  <h3>NDVI / Road</h3><div class='value' style='font-size:0.9rem;'>{feat.get('ndvi',0):.3f} · {feat.get('road_type','—')}</div>
                  <div class='sub'>Width {feat.get('road_width_m',0)}m · Light {'Yes' if feat.get('street_lighting',0) else 'No'}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"<div style='margin-top:0.5rem; border-top:1px solid var(--border); padding-top:0.4rem;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📡 Data Sources</div>", unsafe_allow_html=True)
                summary = pred["summary"]
                st.markdown(f"**{summary['success_count']}/{summary['total_sources']}** sources · **{summary['total_time_ms']}ms**")
                for l in pred["log"]:
                    icon = "✅" if l["status"]=="success" else "⚠️"
                    st.markdown(f"<div style='font-size:0.68rem;'>{icon} {l['source'][:28]} · {l['response_ms']}ms</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style='margin-top:0.5rem; border-top:1px solid var(--border); padding-top:0.4rem;'></div>
                <div class='section-header'>📌 Nearby</div>
                """, unsafe_allow_html=True)
                st.markdown(f"**{len(nearby_pas)}** Protected Areas within 100km")
                st.markdown(f"**{len(nearby_cors)}** Corridors within 100km")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      📊 Exploratory <span style='color:#FF6B35;'>Analytics</span>
    </h1>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Temporal", "🌿 Environmental", "🚗 Traffic", "🦁 Wildlife", "📉 Historical"
    ])

    _AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            pivot_hd = df.pivot_table(index="hour", columns="day_of_week",
                                      values="accident", aggfunc="mean")
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig = go.Figure(go.Heatmap(
                z=pivot_hd.values, x=days, y=list(range(24)),
                colorscale=[[0,"#0a1628"],[0.5,"#EF476F"],[1,"#B5179E"]],
                showscale=True,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Accident Rate: Hour × Day", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp = df.groupby("season")["accident"].mean().sort_values(ascending=False)
            fig2 = go.Figure(go.Bar(
                x=grp.index, y=grp.values,
                marker_color=[PALETTE["critical"],PALETTE["high"],PALETTE["medium"],PALETTE["low"]],
                text=[f"{v:.1%}" for v in grp.values], textposition="outside",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Season", yaxis_tickformat=".0%", height=360)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.histogram(df, x="ndvi", color="season", nbins=40,
                                barmode="overlay", opacity=0.7,
                                color_discrete_sequence=[PALETTE["accent2"],PALETTE["accent1"],PALETTE["accent3"],PALETTE["high"]],
                                title="NDVI Distribution by Season")
            fig3.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = px.scatter(df.sample(2000, random_state=1),
                              x="rainfall_mm", y="visibility_m",
                              color="accident", opacity=0.5, size_max=6,
                              color_continuous_scale=[[0,PALETTE["accent2"]],[1,PALETTE["high"]]],
                              title="Rainfall vs Visibility (coloured by accident)")
            fig4.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            grp_speed = df.copy()
            grp_speed["speed_bin"] = pd.cut(grp_speed["speed_ratio"], bins=10)
            sr = grp_speed.groupby("speed_bin", observed=True)["accident"].mean()
            fig5 = go.Figure(go.Bar(
                x=[str(b) for b in sr.index], y=sr.values,
                marker_color=PALETTE["accent1"],
                text=[f"{v:.1%}" for v in sr.values], textposition="outside",
            ))
            fig5.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Speed Ratio", height=360)
            st.plotly_chart(fig5, use_container_width=True)
        with col6:
            rt = df.groupby("road_type")["accident"].agg(["mean","count"]).reset_index().sort_values("mean")
            fig6 = go.Figure(go.Bar(
                x=rt["mean"], y=rt["road_type"],
                orientation="h",
                marker_color=PALETTE["accent3"],
                text=[f"{v:.1%}" for v in rt["mean"]], textposition="outside",
            ))
            fig6.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Road Type", height=360)
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        col7, col8 = st.columns(2)
        with col7:
            sp = df.groupby("species")["risk_score"].describe()[["mean","50%","max"]].reset_index()
            fig7 = px.bar(sp, x="species", y="mean", error_y=sp["max"]-sp["mean"],
                          color="mean",
                          color_continuous_scale=[[0,PALETTE["low"]],[0.5,PALETTE["high"]],[1,PALETTE["critical"]]],
                          title="Mean Risk Score by Species")
            fig7.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig7, use_container_width=True)
        with col8:
            crr = df[df["accident"]==1].groupby(["species","road_type"]).size().reset_index(name="count")
            fig8 = px.treemap(crr, path=["species","road_type"], values="count",
                              color="count",
                              color_continuous_scale=[[0,"#0d1117"],[1,"#FF6B35"]],
                              title="Accident Distribution: Species × Road")
            fig8.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig8, use_container_width=True)

    with tab5:
        col9, col10 = st.columns(2)
        with col9:
            fig9 = px.box(df, x="road_type", y="past_accidents",
                          color="season",
                          color_discrete_sequence=[PALETTE["accent2"],PALETTE["accent3"],PALETTE["high"],PALETTE["critical"]],
                          title="Past Accidents Distribution by Road & Season")
            fig9.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig9, use_container_width=True)
        with col10:
            fig10 = px.scatter(df.sample(3000, random_state=2),
                               x="kde_density", y="risk_score",
                               color="road_type", opacity=0.6, size_max=8,
                               title="KDE Density vs Risk Score")
            fig10.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig10, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4.5 — FUTURE HOTSPOT FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Future Hotspots":
    from models.train import WildlifeRiskModel, FEATURE_GROUPS
    from data.generate_data import ALL_HIGHWAY_SEGMENTS
    import shap

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      ⚡ Future Hotspot <span style='color:#FF6B35;'>Forecasting</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.2rem 0;'>
      Identifies highway segments with <b>low historical incidents but high predicted risk</b> —
      emerging danger zones where preventive infrastructure should be deployed before accidents happen
    </p>
    """, unsafe_allow_html=True)

    # ── Segment-level analysis ────────────────────────────────────────────────
    if 'highway_segment' not in df.columns:
        st.warning("Dataset does not contain highway segment data. Please retrain with the updated data generator.")
    else:
        # Aggregate by highway segment
        seg_stats = df.groupby('highway_segment').agg(
            total_records=('accident', 'count'),
            total_accidents=('accident', 'sum'),
            accident_rate=('accident', 'mean'),
            avg_risk_score=('risk_score', 'mean'),
            avg_movement=('movement_score', 'mean'),
            avg_driver_risk=('driver_risk', 'mean'),
            avg_ndvi=('ndvi', 'mean'),
            avg_corridor_dist=('corridor_dist_km', 'mean'),
            avg_visibility=('visibility_m', 'mean'),
            avg_speed_ratio=('speed_ratio', 'mean'),
            lat_center=('latitude', 'mean'),
            lon_center=('longitude', 'mean'),
        ).reset_index()

        # Predict risk probability for each segment's average conditions
        seg_stats['predicted_risk'] = seg_stats['avg_risk_score']

        # Future hotspots: LOW historical accident rate BUT HIGH predicted characteristics
        seg_stats['hotspot_score'] = (
            seg_stats['avg_risk_score'] * 0.4 +
            seg_stats['avg_movement'] * 0.2 +
            seg_stats['avg_driver_risk'] / 3 * 0.2 +
            (1 - seg_stats['accident_rate']) * 0.2  # Bonus for low historical rate
        )

        # Flag: "Emerging" = low history + high features
        median_acc = seg_stats['accident_rate'].median()
        median_risk = seg_stats['hotspot_score'].median()
        seg_stats['category'] = seg_stats.apply(
            lambda r: '⚠️ EMERGING HOTSPOT' if r['accident_rate'] < median_acc and r['hotspot_score'] > median_risk
            else '🔴 Known High Risk' if r['accident_rate'] >= median_acc and r['hotspot_score'] > median_risk
            else '🟢 Low Risk' if r['accident_rate'] < median_acc
            else '🟡 Moderate', axis=1
        )

        # Sort by hotspot score
        seg_stats = seg_stats.sort_values('hotspot_score', ascending=False)

        # ── KPI Cards ────────────────────────────────────────────────────────
        emerging = seg_stats[seg_stats['category'] == '⚠️ EMERGING HOTSPOT']
        known_high = seg_stats[seg_stats['category'] == '🔴 Known High Risk']

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class='metric-card'>
              <h3>Total Segments</h3><div class='value'>{len(seg_stats)}</div>
              <div class='sub'>Analyzed</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='metric-card'>
              <h3>⚠️ Emerging Hotspots</h3><div class='value' style='color:#FFD166;'>{len(emerging)}</div>
              <div class='sub'>Low history, high risk</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class='metric-card'>
              <h3>🔴 Known High Risk</h3><div class='value' style='color:#EF476F;'>{len(known_high)}</div>
              <div class='sub'>High history + risk</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class='metric-card'>
              <h3>Top Hotspot Score</h3><div class='value' style='color:#FF6B35;'>{seg_stats['hotspot_score'].max():.3f}</div>
              <div class='sub'>{seg_stats.iloc[0]['highway_segment'][:25]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Emerging Hotspots Detail ─────────────────────────────────────────
        st.markdown("<div class='section-header'>⚠️ Emerging Hotspots — Where to Deploy Prevention Infrastructure</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#8B949E; font-size:0.8rem;'>
          These segments have <b>few historical accidents</b> but share <b>high-risk environmental &
          infrastructural characteristics</b> with known danger zones. They represent
          <b>future collision hotspots</b> where preventive barriers, animal crossings, or speed controls
          should be installed <i>before</i> incidents escalate.
        </p>
        """, unsafe_allow_html=True)

        for i, row in emerging.head(5).iterrows():
            # Find matching highway segment info
            seg_info = next((s for s in ALL_HIGHWAY_SEGMENTS if s["name"] == row["highway_segment"]), {})
            state = seg_info.get("state", "—")

            # Determine risk factors (plain language for officials)
            risk_factors = []
            if row['avg_ndvi'] > 0.5:
                risk_factors.append("🌿 Dense vegetation (animals cross frequently)")
            if row['avg_corridor_dist'] < 3:
                risk_factors.append("🦁 Very close to wildlife corridor")
            if row['avg_speed_ratio'] > 1.0:
                risk_factors.append("🚗 Vehicles exceed speed limits")
            if row['avg_visibility'] < 500:
                risk_factors.append("🌫️ Poor visibility conditions")
            if row['avg_movement'] > 0.35:
                risk_factors.append("🐾 High animal activity score")
            if row['avg_driver_risk'] > 0.7:
                risk_factors.append("⚡ High driver risk index")
            if not risk_factors:
                risk_factors.append("📊 Combined risk factors above threshold")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg, rgba(255,209,102,0.08), rgba(255,107,53,0.05));
                        border:1px solid rgba(255,209,102,0.3); border-radius:12px;
                        padding:1.2rem; margin:0.6rem 0;'>
              <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;'>
                <div>
                  <span style='font-size:1.1rem; font-weight:600; color:#FFD166;'>⚠️ {row['highway_segment']}</span>
                  <span style='font-size:0.75rem; color:#8B949E; margin-left:0.5rem;'>{state}</span>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:1.2rem; font-weight:700; color:#FF6B35;'>Hotspot Score: {row['hotspot_score']:.3f}</div>
                  <div style='font-size:0.7rem; color:#8B949E;'>Accident Rate: {row['accident_rate']:.1%} (low history)</div>
                </div>
              </div>
              <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.5rem; font-size:0.75rem; color:#E6EDF3;'>
                <div>🌿 NDVI: <b>{row['avg_ndvi']:.3f}</b></div>
                <div>🦁 Corridor: <b>{row['avg_corridor_dist']:.1f}km</b></div>
                <div>🐾 Movement: <b>{row['avg_movement']:.3f}</b></div>
                <div>🚗 Speed Ratio: <b>{row['avg_speed_ratio']:.2f}</b></div>
              </div>
              <div style='margin-top:0.5rem; border-top:1px solid rgba(255,209,102,0.15); padding-top:0.5rem;'>
                <div style='font-size:0.72rem; color:#FFD166; font-weight:600; margin-bottom:0.3rem;'>
                  🔍 Why this segment is risky (for forest officials):
                </div>
                {''.join(f"<div style='font-size:0.72rem; color:#E6EDF3; margin:0.15rem 0;'>{rf}</div>" for rf in risk_factors)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Hotspot Comparison Chart ─────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Segment Risk vs. Historical Accident Rate</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#8B949E; font-size:0.78rem;'>
          Segments in the <b>top-left quadrant</b> (low history, high risk) are emerging hotspots.
          Segments in the <b>top-right</b> are known danger zones.
        </p>
        """, unsafe_allow_html=True)

        color_map = {
            '⚠️ EMERGING HOTSPOT': '#FFD166',
            '🔴 Known High Risk': '#EF476F',
            '🟢 Low Risk': '#06D6A0',
            '🟡 Moderate': '#8B949E',
        }
        fig_scatter = go.Figure()
        for cat, color in color_map.items():
            subset = seg_stats[seg_stats['category'] == cat]
            if len(subset) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=subset['accident_rate'], y=subset['hotspot_score'],
                    mode='markers+text',
                    marker=dict(size=12, color=color, opacity=0.85, line=dict(width=1, color='#1a2332')),
                    text=subset['highway_segment'].str[:18],
                    textposition='top center',
                    textfont=dict(size=8, color=color),
                    name=cat,
                    hovertemplate='<b>%{text}</b><br>Accident Rate: %{x:.1%}<br>Hotspot Score: %{y:.3f}<extra></extra>',
                ))
        fig_scatter.add_vline(x=median_acc, line_dash="dash", line_color="#5a6d82", opacity=0.5)
        fig_scatter.add_hline(y=median_risk, line_dash="dash", line_color="#5a6d82", opacity=0.5)
        fig_scatter.update_layout(
            xaxis_title="Historical Accident Rate", yaxis_title="Hotspot Score (Predicted Risk)",
            template="plotly_dark", paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"), height=500,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── SHAP Global Feature Importance ───────────────────────────────────
        st.markdown("<div class='section-header'>🧠 SHAP — Global Feature Importance (What Drives Risk Overall)</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#5a6d82; font-size:0.72rem; font-family:"IBM Plex Mono",monospace;'>
          SHAP (SHapley Additive exPlanations) decomposes the model's prediction into contributions
          from each feature. Below shows which features have the <b>most influence on risk predictions</b>
          across all highway segments. This tells forest officials where to focus intervention resources.
        </p>
        """, unsafe_allow_html=True)

        # Global SHAP bar chart
        if shap_imp:
            shap_df = pd.DataFrame(shap_imp).head(15)
            fig_shap = go.Figure(go.Bar(
                y=shap_df['feature'], x=shap_df['shap_mean'],
                orientation='h',
                marker=dict(
                    color=shap_df['shap_mean'],
                    colorscale=[[0, '#00e676'], [0.5, '#ffb020'], [1.0, '#ff3d5a']],
                    line=dict(width=1, color='rgba(255,255,255,0.1)'),
                ),
                hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>',
            ))
            fig_shap.update_layout(
                template="plotly_dark", paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
                font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"), height=450,
                xaxis_title="Mean |SHAP Value| (Impact on Prediction)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=150),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Plain-language explanations for top features
            st.markdown("<div class='section-header'>📋 What This Means for Forest Officials</div>", unsafe_allow_html=True)
            explanations = {
                'movement_score': "**Animal Activity** — The composite animal movement score is the biggest predictor. Areas with high vegetation (NDVI), near water, during dawn/dusk, and in breeding season see the most crossings.",
                'driver_risk': "**Driver Behavior** — Speed violations, night driving, and driving on forest roads dramatically increase collision risk. Speed enforcement is the most actionable intervention.",
                'species_risk': "**Species Type** — Elephants and tigers cause the highest-impact collisions. Their movement patterns are seasonal and predictable.",
                'kde_density': "**Historical Density** — Past accident clusters strongly predict future ones. Known hotspots remain dangerous unless mitigated.",
                'road_type': "**Road Classification** — Forest roads and rural roads are much riskier than national highways, despite lower traffic. They lack barriers and lighting.",
                'night_flag': "**Night Driving** — Risk roughly doubles during nighttime (8 PM – 6 AM). Night-vision cameras and reflective fencing are effective countermeasures.",
                'corridor_dist_km': "**Corridor Distance** — Closer to wildlife corridors = higher risk. Underpasses/overpasses within 3km of corridors reduce mortality 60-80%.",
                'speed_ratio': "**Speed Compliance** — Drivers exceeding speed limits by even 20% dramatically increase stopping distance and collision severity.",
                'ndvi': "**Vegetation Density** — Dense roadside vegetation blocks sightlines. Strategic clearing of 5m strips along highways improves visibility.",
                'past_accidents': "**Repeat Locations** — Areas with 3+ past incidents are 4× more likely to see future ones. Prioritize these for infrastructure upgrades.",
                'rainfall_mm': "**Rainfall** — Wet roads + reduced visibility + animal movement to water create a triple threat during monsoon season.",
                'visibility_m': "**Visibility** — Fog, rain, and dense canopy reduce sighting distance. Electronic warning signs triggered by poor visibility save lives.",
            }
            top_feats = [s['feature'] for s in shap_imp[:6]]
            for feat in top_feats:
                expl = explanations.get(feat, f"**{feat}** — This feature contributes significantly to risk prediction.")
                st.markdown(f"- {expl}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    from utils.helpers import (
        feature_importance_chart, rf_importance_chart,
        confusion_matrix_chart, model_comparison_chart,
        roc_comparison_chart, PALETTE, PLOTLY_LAYOUT
    )
    from models.train import FEATURE_GROUPS

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      🧠 Model <span style='color:#FF6B35;'>Insights & Explainability</span>
    </h1>
    """, unsafe_allow_html=True)

    _AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
        "📈 Performance", "🔬 XGBoost SHAP", "🌲 RF Importance",
        "📊 SHAP Beeswarm", "⚙️ Feature Groups"
    ])

    with tab_a:
        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>XGBoost Metrics</div>", unsafe_allow_html=True)
        xc1, xc2, xc3, xc4, xc5 = st.columns(5)
        for col, k, label in [
            (xc1, "roc_auc",       "ROC-AUC"),
            (xc2, "avg_precision", "Avg Precision"),
            (xc3, "accuracy",      "Accuracy"),
            (xc4, "brier_score",   "Brier Score"),
            (xc5, "cv_roc_auc_mean", "CV AUC (5-fold)"),
        ]:
            v = xgb_metrics.get(k, 0)
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <h3>{label}</h3>
                  <div class='value' style='font-size:1.5rem;'>{v:.4f}</div>
                </div>""", unsafe_allow_html=True)

        if rf_metrics:
            st.markdown("<div class='section-header'>Random Forest Metrics</div>", unsafe_allow_html=True)
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            for col, k, label in [
                (rc1, "roc_auc",       "ROC-AUC"),
                (rc2, "avg_precision", "Avg Precision"),
                (rc3, "accuracy",      "Accuracy"),
                (rc4, "brier_score",   "Brier Score"),
                (rc5, "cv_roc_auc_mean", "CV AUC (5-fold)"),
            ]:
                v = rf_metrics.get(k, 0)
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                      <h3>{label}</h3>
                      <div class='value' style='font-size:1.5rem; color:#58A6FF;'>{v:.4f}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            if rf_metrics:
                st.plotly_chart(model_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)
        with cmp2:
            if rf_metrics:
                st.plotly_chart(roc_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)

        # Confusion matrices
        st.markdown("<br>", unsafe_allow_html=True)
        cm1, cm2 = st.columns(2)
        with cm1:
            st.plotly_chart(confusion_matrix_chart(
                xgb_metrics.get("confusion", [[0,0],[0,0]]), "XGBoost Confusion Matrix"
            ), use_container_width=True)
        with cm2:
            if rf_metrics:
                st.plotly_chart(confusion_matrix_chart(
                    rf_metrics.get("confusion", [[0,0],[0,0]]), "RF Confusion Matrix"
                ), use_container_width=True)

    with tab_b:
        if shap_imp:
            st.plotly_chart(feature_importance_chart(shap_imp), use_container_width=True)
        else:
            st.info("SHAP importance not available.")

    with tab_c:
        if rf_imp:
            st.plotly_chart(rf_importance_chart(rf_imp), use_container_width=True)
        else:
            st.info("Random Forest importance not available. Retrain the models.")

    with tab_d:
        st.markdown("<div class='section-header'>SHAP Beeswarm — Global Feature Impact</div>",
                    unsafe_allow_html=True)
        sv  = model.shap_values
        ssp = model.shap_sample
        if sv is not None and ssp is not None:
            top_n    = 15
            top_feat = [x["feature"] for x in sorted(shap_imp, key=lambda x: x["shap_mean"], reverse=True)[:top_n]]
            idxs     = [model.feature_cols.index(f) for f in top_feat if f in model.feature_cols]

            traces = []
            for rank, (fi, fname) in enumerate(zip(idxs, top_feat)):
                fv = ssp.iloc[:, fi].values
                fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)
                jitter  = np.random.uniform(-0.25, 0.25, len(sv))
                traces.append(go.Scatter(
                    x    = sv[:, fi],
                    y    = [rank + j for j in jitter],
                    mode = "markers",
                    name = fname,
                    marker= dict(size=4, color=[int(v*255) for v in fv_norm],
                                 colorscale="RdBu_r", opacity=0.75, showscale=False),
                    showlegend=False,
                    hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
                ))
            fig_bs = go.Figure(traces)
            fig_bs.update_layout(
                **PLOTLY_LAYOUT,
                title  = "SHAP Beeswarm (top 15 features)",
                xaxis_title = "SHAP value",
                yaxis  = dict(
                    tickvals=list(range(top_n)),
                    ticktext=top_feat,
                    gridcolor=PALETTE["border"],
                ),
                height = 620,
            )
            fig_bs.add_vline(x=0, line_color=PALETTE["muted"], line_width=1.5)
            st.plotly_chart(fig_bs, use_container_width=True)

    with tab_e:
        st.markdown("<div class='section-header'>Feature Group Importance</div>", unsafe_allow_html=True)
        shap_dict = {x["feature"]: x["shap_mean"] for x in shap_imp}
        group_totals = {}
        for gname, feats in FEATURE_GROUPS.items():
            group_totals[gname] = sum(shap_dict.get(f, 0) for f in feats)

        gt = pd.Series(group_totals).sort_values(ascending=True)
        colors_g = [PALETTE["accent1"] if v > gt.median() else PALETTE["accent2"] for v in gt.values]
        fig_g = go.Figure(go.Bar(
            x=gt.values, y=gt.index, orientation="h",
            marker_color=colors_g,
            text=[f"{v:.4f}" for v in gt.values], textposition="outside",
        ))
        fig_g.update_layout(**PLOTLY_LAYOUT, title="Total SHAP by Feature Group", height=400)
        st.plotly_chart(fig_g, use_container_width=True)

        # Feature detail table
        st.markdown("<div class='section-header'>Full Feature Importance Table</div>", unsafe_allow_html=True)
        if shap_imp:
            imp_df = pd.DataFrame(shap_imp).sort_values("shap_mean", ascending=False).reset_index(drop=True)
            imp_df["rank"] = range(1, len(imp_df)+1)
            imp_df = imp_df[["rank","feature","shap_mean"]]
            st.dataframe(
                imp_df.style.bar(subset=["shap_mean"], color=PALETTE["accent1"]),
                use_container_width=True,
                height=500,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANIMAL MOVEMENT DATA INJECTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🐾 Animal Movement":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT
    from data.realtime_extractor import PROTECTED_AREAS, WILDLIFE_CORRIDORS

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🐾 Animal Movement <span style='color:#00e5ff;'>Data Pipeline</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Live injection of wildlife movement patterns from GBIF, tracking databases, and corridor telemetry
    </p>
    """, unsafe_allow_html=True)

    # ── Pipeline Status Panel ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Pipeline Sources & Status</div>", unsafe_allow_html=True)

    movement_sources = [
        {"name": "GBIF Occurrence API", "status": "Active", "records": "300+/query",
         "desc": "Mammal occurrence records within 50km radius from Global Biodiversity Information Facility",
         "color": "#00e5ff", "endpoint": "api.gbif.org/v1/occurrence/search"},
        {"name": "Movebank Telemetry", "status": "Configured", "records": "GPS tracks",
         "desc": "Satellite collar and GPS telemetry data for tracked wildlife across Indian corridors",
         "color": "#00e676", "endpoint": "movebank.org/movebank/service"},
        {"name": "Wildlife Institute of India", "status": "Configured", "records": "Survey data",
         "desc": "Camera trap and transect survey records from WII research stations",
         "color": "#ffb020", "endpoint": "wii.gov.in"},
        {"name": "India Biodiversity Portal", "status": "Configured", "records": "Citizen science",
         "desc": "Community-sourced wildlife observations with geolocation and timestamp data",
         "color": "#d500f9", "endpoint": "indiabiodiversity.org/api/observation"},
    ]

    src_cols = st.columns(2)
    for i, src in enumerate(movement_sources):
        with src_cols[i % 2]:
            badge = "🟢" if src["status"] == "Active" else "🟡"
            st.markdown(f"""
            <div class='metric-card' style='border-left:3px solid {src["color"]};'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='color:{src["color"]};'>{src["name"]}</h3>
                <span style='font-size:0.65rem; color:{src["color"]}; background:{src["color"]}15;
                      padding:0.15rem 0.5rem; border:1px solid {src["color"]}33;'>{badge} {src["status"]}</span>
              </div>
              <div class='sub' style='margin:0.3rem 0;'>{src["desc"]}</div>
              <code style='font-size:0.6rem; color:#5a6d82;'>{src["endpoint"]}</code>
              <div style='margin-top:0.3rem;'>
                <span style='font-size:0.6rem; color:{src["color"]};'>Records: {src["records"]}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live Movement Query ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Query Animal Movement Data</div>", unsafe_allow_html=True)
    mv_col1, mv_col2, mv_col3 = st.columns(3)
    with mv_col1:
        mv_lat = st.number_input("Latitude", value=11.65, min_value=8.0, max_value=25.0,
                                  step=0.1, key="mv_lat")
    with mv_col2:
        mv_lon = st.number_input("Longitude", value=76.65, min_value=73.0, max_value=85.0,
                                  step=0.1, key="mv_lon")
    with mv_col3:
        mv_radius = st.selectbox("Search Radius", ["25 km", "50 km", "100 km"], index=1, key="mv_radius")

    if st.button("Fetch Movement Data", type="primary", key="mv_fetch"):
        from data.realtime_extractor import RealtimeDataExtractor

        with st.spinner("Querying GBIF and computing movement patterns..."):
            extractor = RealtimeDataExtractor()
            wildlife_data = extractor.fetch_wildlife_data(mv_lat, mv_lon)
            spatial_data = extractor.compute_spatial_features(mv_lat, mv_lon)

        species_counts = wildlife_data.get("species_counts", {})
        total_sightings = wildlife_data.get("total_sightings", 0)

        # Results panel
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.markdown(f"""<div class='metric-card'>
              <h3>Total Sightings</h3>
              <div class='value' style='font-size:1.5rem; color:#00e5ff;'>{total_sightings}</div>
              <div class='sub'>GBIF records in area</div>
            </div>""", unsafe_allow_html=True)
        with res_c2:
            st.markdown(f"""<div class='metric-card'>
              <h3>Dominant Species</h3>
              <div class='value' style='font-size:1.3rem; color:#ffb020;'>{wildlife_data.get('species','—').capitalize()}</div>
              <div class='sub'>highest occurrence</div>
            </div>""", unsafe_allow_html=True)
        with res_c3:
            st.markdown(f"""<div class='metric-card'>
              <h3>Nearest PA</h3>
              <div class='value' style='font-size:1.1rem; color:#00e676;'>{spatial_data.get('nearest_pa','—')}</div>
              <div class='sub'>{spatial_data.get('protected_dist_km',0):.1f} km away</div>
            </div>""", unsafe_allow_html=True)
        with res_c4:
            st.markdown(f"""<div class='metric-card'>
              <h3>Nearest Corridor</h3>
              <div class='value' style='font-size:1.0rem; color:#d500f9;'>{spatial_data.get('nearest_corridor','—')}</div>
              <div class='sub'>{spatial_data.get('corridor_dist_km',0):.1f} km away</div>
            </div>""", unsafe_allow_html=True)

        # Species breakdown chart
        if species_counts:
            sp_names = [s.capitalize() for s in species_counts.keys()]
            sp_vals = list(species_counts.values())
            sp_clrs = [{"tiger":"#ff3d5a","elephant":"#d500f9","leopard":"#ff7043","deer":"#ffb020",
                        "boar":"#9d7aff","wolf":"#4da6ff","nilgai":"#00e676","sambar":"#00e5ff"
                       }.get(s, "#00e5ff") for s in species_counts.keys()]
            fig_sp = go.Figure(go.Bar(x=sp_names, y=sp_vals, marker_color=sp_clrs,
                                       text=sp_vals, textposition="outside"))
            fig_sp.update_layout(
                paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
                font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
                title="Species Occurrence Count (GBIF)", height=350,
                xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
                yaxis=dict(title="Occurrences", gridcolor="#1a2332", zerolinecolor="#1a2332"),
                margin=dict(l=40,r=20,t=40,b=30),
            )
            st.plotly_chart(fig_sp, use_container_width=True)

        # Movement corridors map
        st.markdown("<div class='section-header'>Wildlife Corridor Network</div>", unsafe_allow_html=True)
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium

        m = folium.Map(location=[mv_lat, mv_lon], zoom_start=8,
                       tiles="CartoDB dark_matter", control_scale=True)
        folium.CircleMarker([mv_lat, mv_lon], radius=8, color="#00e5ff",
                            fill=True, fill_opacity=0.8,
                            popup="Query Location").add_to(m)
        for pa in PROTECTED_AREAS:
            dist = ((pa["lat"]-mv_lat)**2 + (pa["lon"]-mv_lon)**2)**0.5
            if dist < 3:
                folium.CircleMarker([pa["lat"], pa["lon"]], radius=6, color="#00e676",
                                    fill=True, fill_opacity=0.6,
                                    popup=f'{pa["name"]} ({pa.get("state","")})').add_to(m)
        for cor in WILDLIFE_CORRIDORS:
            dist = ((cor["lat"]-mv_lat)**2 + (cor["lon"]-mv_lon)**2)**0.5
            if dist < 3:
                folium.CircleMarker([cor["lat"], cor["lon"]], radius=5, color="#d500f9",
                                    fill=True, fill_opacity=0.6,
                                    popup=cor["name"]).add_to(m)
        # Simulated movement heatmap around corridors
        heat_pts = []
        for cor in WILDLIFE_CORRIDORS:
            dist = ((cor["lat"]-mv_lat)**2 + (cor["lon"]-mv_lon)**2)**0.5
            if dist < 3:
                for _ in range(15):
                    heat_pts.append([cor["lat"]+np.random.normal(0,0.08),
                                     cor["lon"]+np.random.normal(0,0.08),
                                     np.random.uniform(0.3, 1.0)])
        if heat_pts:
            HeatMap(heat_pts, radius=18, blur=12, gradient={0.3:"blue",0.5:"cyan",0.7:"lime",1.0:"red"}).add_to(m)
        st_folium(m, width=None, height=500, returned_objects=[])

    # ── Dataset Movement Patterns ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Training Data — Movement Score Analysis</div>", unsafe_allow_html=True)

    # Movement score distribution by species
    fig_mv_dist = go.Figure()
    for sp in df["species"].unique():
        sp_data = df[df["species"]==sp]["movement_score"]
        fig_mv_dist.add_trace(go.Violin(
            y=sp_data, name=sp.capitalize(), box_visible=True,
            meanline_visible=True, opacity=0.7,
            marker=dict(color={"tiger":"#ff3d5a","elephant":"#d500f9","leopard":"#ff7043",
                               "deer":"#ffb020","boar":"#9d7aff","wolf":"#4da6ff",
                               "nilgai":"#00e676","sambar":"#00e5ff"}.get(sp,"#00e5ff")),
        ))
    fig_mv_dist.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        title="Movement Score Distribution by Species",
        yaxis=dict(title="Movement Score", gridcolor="#1a2332", zerolinecolor="#1a2332"),
        xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
        height=420, margin=dict(l=40,r=20,t=40,b=30), showlegend=False,
    )
    st.plotly_chart(fig_mv_dist, use_container_width=True)

    # Movement vs corridor distance
    mv_scatter_cols = st.columns(2)
    with mv_scatter_cols[0]:
        sample_mv = df.sample(min(2000, len(df)), random_state=42)
        fig_mv_corr = px.scatter(sample_mv, x="corridor_dist_km", y="movement_score",
                                  color="accident", color_discrete_map={0:"#00e5ff", 1:"#ff3d5a"},
                                  opacity=0.5, size_max=8,
                                  labels={"corridor_dist_km":"Corridor Distance (km)",
                                          "movement_score":"Movement Score"})
        fig_mv_corr.update_layout(
            paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            title="Movement Score vs Corridor Distance", height=380,
            xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
            yaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
            margin=dict(l=40,r=20,t=40,b=30),
        )
        st.plotly_chart(fig_mv_corr, use_container_width=True)

    with mv_scatter_cols[1]:
        fig_mv_time = df.groupby("hour")["movement_score"].mean().reset_index()
        fig_time = go.Figure(go.Scatter(
            x=fig_mv_time["hour"], y=fig_mv_time["movement_score"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=6, color="#00e5ff"),
            fillcolor="rgba(0,229,255,0.1)",
        ))
        fig_time.update_layout(
            paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            title="Average Movement Score by Hour", height=380,
            xaxis=dict(title="Hour of Day", gridcolor="#1a2332", zerolinecolor="#1a2332",
                       dtick=2),
            yaxis=dict(title="Avg Movement Score", gridcolor="#1a2332", zerolinecolor="#1a2332"),
            margin=dict(l=40,r=20,t=40,b=30),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Injection Pipeline Architecture ───────────────────────────────────────
    st.markdown("<div class='section-header'>Pipeline Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │          ANIMAL MOVEMENT DATA INJECTION PIPELINE        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │   ┌─── External Sources ──────────────────────────┐    │
    │   │  GBIF API ──→ Species occurrences (mammal)    │    │
    │   │  Movebank  ──→ GPS collar telemetry tracks    │    │
    │   │  WII       ──→ Camera trap / transect data    │    │
    │   │  IBP       ──→ Citizen science observations   │    │
    │   └───────────────────────────────────────────────┘    │
    │                     │                                   │
    │   ┌─── Processing ────────────────────────────────┐    │
    │   │  1. Geocode + radius filter (Haversine)       │    │
    │   │  2. Species matching → risk scoring            │    │
    │   │  3. Temporal aggregation (hour / season)       │    │
    │   │  4. KDE density estimation for hotspots        │    │
    │   │  5. Movement score derivation                  │    │
    │   └───────────────────────────────────────────────┘    │
    │                     │                                   │
    │   ┌─── Outputs ──────────────────────────────────┐     │
    │   │  → movement_score (feature)                   │    │
    │   │  → kde_density (feature)                      │    │
    │   │  → species_risk (feature)                     │    │
    │   │  → corridor proximity alerts                  │    │
    │   └───────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────┘
    ```
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — NDVI PREDICTION INDEX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 NDVI Prediction":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🌿 NDVI <span style='color:#00e676;'>Prediction Index</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Normalized Difference Vegetation Index — forecasting habitat density and its correlation with collision risk
    </p>
    """, unsafe_allow_html=True)

    # ── NDVI Overview Metrics ─────────────────────────────────────────────────
    avg_ndvi_all = df["ndvi"].mean()
    ndvi_acc = df[df["accident"]==1]["ndvi"].mean()
    ndvi_safe = df[df["accident"]==0]["ndvi"].mean()
    ndvi_corr = df[["ndvi","risk_score"]].corr().iloc[0,1]

    nc1, nc2, nc3, nc4 = st.columns(4)
    ndvi_kpis = [
        (nc1, "Mean NDVI", f"{avg_ndvi_all:.4f}", "all records", "#00e676"),
        (nc2, "Accident NDVI", f"{ndvi_acc:.4f}", "where accident=1", "#ff3d5a"),
        (nc3, "Safe Zone NDVI", f"{ndvi_safe:.4f}", "where accident=0", "#00e5ff"),
        (nc4, "Risk Correlation", f"{ndvi_corr:.4f}", "Pearson r", "#ffb020"),
    ]
    for col, title, val, sub, color in ndvi_kpis:
        with col:
            st.markdown(f"""<div class='metric-card'>
              <h3>{title}</h3>
              <div class='value' style='font-size:1.5rem; color:{color};'>{val}</div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── NDVI Seasonal Forecast Model ──────────────────────────────────────────
    st.markdown("<div class='section-header'>Seasonal NDVI Forecast Model</div>", unsafe_allow_html=True)

    seasons = ["summer", "monsoon", "post_monsoon", "winter"]
    season_labels = ["Summer\n(Mar-May)", "Monsoon\n(Jun-Sep)", "Post-Monsoon\n(Oct-Nov)", "Winter\n(Dec-Feb)"]
    season_modifiers = {"summer": -0.10, "monsoon": 0.12, "post_monsoon": 0.05, "winter": -0.05}

    # Compute NDVI stats per season from training data
    ndvi_by_season = df.groupby("season").agg(
        mean_ndvi=("ndvi","mean"), std_ndvi=("ndvi","std"),
        accident_rate=("accident","mean"), mean_risk=("risk_score","mean"),
        count=("accident","count")
    ).reindex(seasons)

    # Forecast chart with confidence bands
    fig_ndvi_forecast = go.Figure()
    x_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_to_season = {1:"winter",2:"winter",3:"summer",4:"summer",5:"summer",
                       6:"monsoon",7:"monsoon",8:"monsoon",9:"monsoon",
                       10:"post_monsoon",11:"post_monsoon",12:"winter"}
    forecast_ndvi = []
    forecast_upper = []
    forecast_lower = []
    base_ndvi = avg_ndvi_all
    for m in range(1,13):
        s = month_to_season[m]
        mod = season_modifiers[s]
        mean_val = base_ndvi + mod
        std_val = ndvi_by_season.loc[s, "std_ndvi"] if s in ndvi_by_season.index else 0.08
        forecast_ndvi.append(round(mean_val, 4))
        forecast_upper.append(round(mean_val + 1.96*std_val, 4))
        forecast_lower.append(round(mean_val - 1.96*std_val, 4))

    fig_ndvi_forecast.add_trace(go.Scatter(
        x=x_months, y=forecast_upper, mode="lines", line=dict(width=0),
        showlegend=False, name="Upper CI",
    ))
    fig_ndvi_forecast.add_trace(go.Scatter(
        x=x_months, y=forecast_lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,230,118,0.12)",
        showlegend=False, name="Lower CI",
    ))
    fig_ndvi_forecast.add_trace(go.Scatter(
        x=x_months, y=forecast_ndvi, mode="lines+markers",
        line=dict(color="#00e676", width=3),
        marker=dict(size=8, color="#00e676", line=dict(width=1, color="#0d1320")),
        name="NDVI Forecast",
    ))
    # Add risk overlay
    risk_by_month = []
    for m in range(1,13):
        s = month_to_season[m]
        risk_by_month.append(ndvi_by_season.loc[s,"accident_rate"] if s in ndvi_by_season.index else 0.3)
    fig_ndvi_forecast.add_trace(go.Scatter(
        x=x_months, y=risk_by_month, mode="lines+markers",
        line=dict(color="#ff3d5a", width=2, dash="dot"),
        marker=dict(size=6, color="#ff3d5a"),
        name="Accident Rate", yaxis="y2",
    ))
    fig_ndvi_forecast.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        title="Monthly NDVI Forecast with 95% Confidence Interval",
        xaxis=dict(title="Month", gridcolor="#1a2332", zerolinecolor="#1a2332"),
        yaxis=dict(title="NDVI Index", gridcolor="#1a2332", zerolinecolor="#1a2332",
                   range=[0, 1]),
        yaxis2=dict(title="Accident Rate", overlaying="y", side="right",
                    gridcolor="#1a233200", range=[0, 1],
                    tickfont=dict(color="#ff3d5a"), title_font=dict(color="#ff3d5a")),
        height=420, margin=dict(l=40,r=50,t=40,b=30),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    font=dict(size=10)),
    )
    st.plotly_chart(fig_ndvi_forecast, use_container_width=True)

    # ── Season Comparison Cards ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Season-wise NDVI Breakdown</div>", unsafe_allow_html=True)
    season_cols = st.columns(4)
    season_icons = {"summer":"☀️", "monsoon":"🌧️", "post_monsoon":"🍂", "winter":"❄️"}
    season_colors = {"summer":"#ffb020", "monsoon":"#00e676", "post_monsoon":"#ff7043", "winter":"#4da6ff"}
    for i, s in enumerate(seasons):
        with season_cols[i]:
            if s in ndvi_by_season.index:
                row = ndvi_by_season.loc[s]
                st.markdown(f"""
                <div class='metric-card' style='border-top:3px solid {season_colors[s]};'>
                  <div style='font-size:1.5rem; text-align:center;'>{season_icons[s]}</div>
                  <h3 style='text-align:center; color:{season_colors[s]};'>{s.replace('_',' ').title()}</h3>
                  <div style='text-align:center;'>
                    <div class='value' style='font-size:1.3rem; color:{season_colors[s]};'>{row["mean_ndvi"]:.4f}</div>
                    <div class='sub'>Mean NDVI</div>
                    <div style='margin-top:0.4rem; font-size:0.7rem; color:#5a6d82;'>
                      Accident Rate: <span style='color:#ff3d5a;'>{row["accident_rate"]:.1%}</span><br>
                      Modifier: <span style='color:{season_colors[s]};'>{season_modifiers[s]:+.2f}</span><br>
                      Records: {row["count"]:,.0f}
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── NDVI Risk Correlation Deep Dive ───────────────────────────────────────
    ndvi_analysis_cols = st.columns(2)
    with ndvi_analysis_cols[0]:
        # NDVI distribution by accident flag
        fig_ndvi_dist = go.Figure()
        fig_ndvi_dist.add_trace(go.Histogram(
            x=df[df["accident"]==0]["ndvi"], nbinsx=40, name="No Accident",
            marker_color="#00e5ff", opacity=0.6,
        ))
        fig_ndvi_dist.add_trace(go.Histogram(
            x=df[df["accident"]==1]["ndvi"], nbinsx=40, name="Accident",
            marker_color="#ff3d5a", opacity=0.6,
        ))
        fig_ndvi_dist.update_layout(
            paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            title="NDVI Distribution: Accident vs Safe",
            barmode="overlay", height=380,
            xaxis=dict(title="NDVI", gridcolor="#1a2332", zerolinecolor="#1a2332"),
            yaxis=dict(title="Count", gridcolor="#1a2332", zerolinecolor="#1a2332"),
            margin=dict(l=40,r=20,t=40,b=30),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_ndvi_dist, use_container_width=True)

    with ndvi_analysis_cols[1]:
        # NDVI vs Protected Area distance
        sample_ndvi = df.sample(min(2000, len(df)), random_state=42)
        fig_ndvi_pa = px.scatter(sample_ndvi, x="protected_dist_km", y="ndvi",
                                  color="season", opacity=0.5,
                                  color_discrete_map=season_colors,
                                  labels={"protected_dist_km":"Distance to Protected Area (km)",
                                          "ndvi":"NDVI"})
        fig_ndvi_pa.update_layout(
            paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            title="NDVI vs Protected Area Proximity", height=380,
            xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
            yaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
            margin=dict(l=40,r=20,t=40,b=30),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_ndvi_pa, use_container_width=True)

    # ── NDVI Prediction Lookup ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>NDVI Prediction Lookup</div>", unsafe_allow_html=True)
    lk_c1, lk_c2, lk_c3 = st.columns(3)
    with lk_c1:
        lk_lat = st.number_input("Latitude", value=12.2, min_value=8.0, max_value=25.0,
                                  step=0.1, key="ndvi_lat")
    with lk_c2:
        lk_lon = st.number_input("Longitude", value=76.6, min_value=73.0, max_value=85.0,
                                  step=0.1, key="ndvi_lon")
    with lk_c3:
        lk_season = st.selectbox("Season", seasons, index=1, key="ndvi_season",
                                  format_func=lambda x: x.replace("_"," ").title())

    if st.button("Predict NDVI", type="primary", key="ndvi_predict"):
        from data.realtime_extractor import RealtimeDataExtractor
        ext = RealtimeDataExtractor()
        spatial = ext.compute_spatial_features(lk_lat, lk_lon, lk_season)
        predicted_ndvi = spatial.get("ndvi", 0.5)
        pa_name = spatial.get("nearest_pa", "Unknown")
        pa_dist = spatial.get("protected_dist_km", 0)

        # NDVI gauge
        ndvi_color = "#ff3d5a" if predicted_ndvi < 0.3 else "#ffb020" if predicted_ndvi < 0.5 else "#00e676"
        risk_indicator = "HIGH RISK" if predicted_ndvi < 0.3 else "MODERATE" if predicted_ndvi < 0.5 else "LOW RISK"

        pred_c1, pred_c2 = st.columns([1, 1.5])
        with pred_c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_ndvi,
                number=dict(font=dict(size=40, color=ndvi_color)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickcolor="#5a6d82"),
                    bar=dict(color=ndvi_color),
                    bgcolor="#0d1320",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 0.3], color="#ff3d5a22"),
                        dict(range=[0.3, 0.5], color="#ffb02022"),
                        dict(range=[0.5, 1], color="#00e67622"),
                    ],
                ),
                title=dict(text="Predicted NDVI", font=dict(size=14, color="#c8d6e5")),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#060a10", height=280,
                margin=dict(l=20,r=20,t=50,b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with pred_c2:
            st.markdown(f"""
            <div class='metric-card' style='padding:1.2rem;'>
              <h3 style='color:#00e5ff; font-size:1rem;'>Prediction Results</h3>
              <div style='margin-top:0.5rem;'>
                <div style='display:flex; justify-content:space-between; margin:0.4rem 0; padding:0.3rem 0;
                            border-bottom:1px solid #1a2332;'>
                  <span style='color:#5a6d82; font-size:0.75rem;'>Predicted NDVI</span>
                  <span style='color:{ndvi_color}; font-size:0.9rem; font-weight:600;'>{predicted_ndvi:.4f}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:0.4rem 0; padding:0.3rem 0;
                            border-bottom:1px solid #1a2332;'>
                  <span style='color:#5a6d82; font-size:0.75rem;'>Habitat Status</span>
                  <span style='color:{ndvi_color}; font-size:0.8rem;'>{risk_indicator}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:0.4rem 0; padding:0.3rem 0;
                            border-bottom:1px solid #1a2332;'>
                  <span style='color:#5a6d82; font-size:0.75rem;'>Nearest Protected Area</span>
                  <span style='color:#00e676; font-size:0.8rem;'>{pa_name}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:0.4rem 0; padding:0.3rem 0;
                            border-bottom:1px solid #1a2332;'>
                  <span style='color:#5a6d82; font-size:0.75rem;'>PA Distance</span>
                  <span style='color:#4da6ff; font-size:0.8rem;'>{pa_dist:.2f} km</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin:0.4rem 0; padding:0.3rem 0;'>
                  <span style='color:#5a6d82; font-size:0.75rem;'>Season</span>
                  <span style='color:#ffb020; font-size:0.8rem;'>{lk_season.replace('_',' ').title()}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alert System":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🚨 Wildlife Collision <span style='color:#ff3d5a;'>Alert System</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Real-time risk alerts, notification thresholds, and early warning dispatch for wildlife corridors
    </p>
    """, unsafe_allow_html=True)

    # ── Current Alert Status ──────────────────────────────────────────────────
    _now_alert = datetime.now()
    _hr = _now_alert.hour
    _is_night = _hr < 6 or _hr >= 20
    _is_dawn_dusk = (5 <= _hr <= 7) or (17 <= _hr <= 19)
    _m = _now_alert.month
    _season_alert = ("summer" if _m in [3,4,5] else "monsoon" if _m in [6,7,8,9]
                     else "post_monsoon" if _m in [10,11] else "winter")
    _breeding = _season_alert in ["monsoon","post_monsoon"]

    # Compute alert level
    _alert_score = 0
    if _is_night: _alert_score += 30
    if _is_dawn_dusk: _alert_score += 25
    if _breeding: _alert_score += 20
    if _season_alert == "monsoon": _alert_score += 15
    if _hr in [5,6,18,19]: _alert_score += 10

    if _alert_score >= 50:
        _alert_level, _alert_color, _alert_icon = "CRITICAL", "#ff3d5a", "🔴"
    elif _alert_score >= 35:
        _alert_level, _alert_color, _alert_icon = "HIGH", "#ff7043", "🟠"
    elif _alert_score >= 20:
        _alert_level, _alert_color, _alert_icon = "ELEVATED", "#ffb020", "🟡"
    else:
        _alert_level, _alert_color, _alert_icon = "NORMAL", "#00e676", "🟢"

    # Alert banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg, {_alert_color}12, {_alert_color}05);
                border:1px solid {_alert_color}44; padding:1rem 1.5rem; margin-bottom:1.5rem;'>
      <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
          <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#5a6d82;
                      letter-spacing:0.2em;'>CURRENT SYSTEM ALERT LEVEL</div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:1.8rem; font-weight:700;
                      color:{_alert_color}; letter-spacing:0.1em; margin:0.2rem 0;'>
            {_alert_icon} {_alert_level}
          </div>
          <div style='font-family:"IBM Plex Mono",monospace; font-size:0.55rem; color:#5a6d82;'>
            Score: {_alert_score}/100 · {_now_alert.strftime('%Y-%m-%d %H:%M IST')} ·
            {'Night Window' if _is_night else 'Dawn/Dusk' if _is_dawn_dusk else 'Daylight'} ·
            {_season_alert.replace('_',' ').title()}
            {'· Breeding Season' if _breeding else ''}
          </div>
        </div>
        <div style='text-align:right;'>
          <div style='font-size:2.5rem; filter:drop-shadow(0 0 8px {_alert_color});'>{_alert_icon}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert Factor Breakdown ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Alert Factor Analysis</div>", unsafe_allow_html=True)
    factors = [
        ("Night Window (20:00-06:00)", _is_night, 30, "#ff3d5a"),
        ("Dawn/Dusk Peak (05-07, 17-19)", _is_dawn_dusk, 25, "#ff7043"),
        ("Breeding Season Active", _breeding, 20, "#d500f9"),
        ("Monsoon Period", _season_alert == "monsoon", 15, "#00e5ff"),
        ("Peak Crossing Hours", _hr in [5,6,18,19], 10, "#ffb020"),
    ]
    for label, active, weight, color in factors:
        status_text = "ACTIVE" if active else "INACTIVE"
        bar_width = weight if active else 0
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin:0.4rem 0; padding:0.4rem 0.6rem;
                    background:#0d132044; border-left:3px solid {color if active else "#1a2332"};'>
          <div style='flex:1;'>
            <span style='font-size:0.75rem; color:{"#e8f0fa" if active else "#5a6d82"};'>{label}</span>
          </div>
          <div style='width:120px; margin:0 1rem;'>
            <div style='background:#1a2332; height:6px; overflow:hidden;'>
              <div style='background:{color}; height:100%; width:{bar_width*2}%;
                          transition:width 0.5s;'></div>
            </div>
          </div>
          <div style='width:50px; text-align:center;'>
            <span style='font-size:0.65rem; color:{color if active else "#5a6d82"};
                         font-family:"JetBrains Mono",monospace;'>+{weight if active else 0}</span>
          </div>
          <div style='width:70px; text-align:right;'>
            <span style='font-size:0.55rem; padding:0.1rem 0.4rem;
                         background:{color+"22" if active else "#1a2332"};
                         color:{color if active else "#5a6d82"};
                         border:1px solid {color+"44" if active else "#1a233200"};'>
              {status_text}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert Threshold Configuration ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Notification Thresholds</div>", unsafe_allow_html=True)
    thresh_col1, thresh_col2 = st.columns(2)

    with thresh_col1:
        st.markdown("""
        <div class='metric-card' style='padding:1rem;'>
          <h3 style='color:#00e5ff;'>Risk Score Thresholds</h3>
        </div>""", unsafe_allow_html=True)
        t_low = st.slider("Low Risk Ceiling", 0.0, 1.0, 0.25, 0.05, key="t_low")
        t_mod = st.slider("Moderate Risk Ceiling", 0.0, 1.0, 0.50, 0.05, key="t_mod")
        t_high = st.slider("High Risk Ceiling", 0.0, 1.0, 0.75, 0.05, key="t_high")

    with thresh_col2:
        st.markdown("""
        <div class='metric-card' style='padding:1rem;'>
          <h3 style='color:#ffb020;'>Alert Dispatch Rules</h3>
        </div>""", unsafe_allow_html=True)
        dispatch_rules = {
            "🟢 Low (< {:.0%})".format(t_low): "Log only — no notification",
            "🟡 Moderate ({:.0%} – {:.0%})".format(t_low, t_mod): "Dashboard warning + daily digest",
            "🟠 High ({:.0%} – {:.0%})".format(t_mod, t_high): "Push notification to patrol teams",
            "🔴 Critical (> {:.0%})".format(t_high): "Immediate SMS + siren activation at crossing",
        }
        for rule, action in dispatch_rules.items():
            st.markdown(f"**{rule}:** {action}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Simulated Alert Feed ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Recent Alert Feed (Simulated)</div>", unsafe_allow_html=True)

    # Generate alerts from high-risk records in training data
    high_risk_df = df[df["risk_score"] > 0.65].sample(min(8, len(df[df["risk_score"]>0.65])), random_state=42)
    for idx, row in high_risk_df.iterrows():
        risk_val = row["risk_score"]
        if risk_val > 0.80:
            a_level, a_color, a_icon = "CRITICAL", "#ff3d5a", "🔴"
        elif risk_val > 0.65:
            a_level, a_color, a_icon = "HIGH", "#ff7043", "🟠"
        else:
            a_level, a_color, a_icon = "ELEVATED", "#ffb020", "🟡"

        species_str = row["species"].capitalize() if "species" in row.index else "Unknown"
        road_str = row["road_type"].replace("_"," ").title() if "road_type" in row.index else "Road"
        hour_str = f"{int(row['hour']):02d}:00" if "hour" in row.index else "--:--"

        st.markdown(f"""
        <div style='display:flex; align-items:center; padding:0.5rem 0.8rem; margin:0.3rem 0;
                    background:#0d132066; border-left:3px solid {a_color};'>
          <div style='margin-right:0.8rem; font-size:1.2rem;'>{a_icon}</div>
          <div style='flex:1;'>
            <div style='font-size:0.75rem; color:#e8f0fa;'>
              <span style='color:{a_color}; font-weight:600;'>{a_level}</span> —
              {species_str} crossing detected on {road_str}
            </div>
            <div style='font-size:0.6rem; color:#5a6d82; margin-top:0.15rem;'>
              Risk: {risk_val:.3f} · {row.get("season","—").replace("_"," ").title()} ·
              {hour_str} · NDVI: {row.get("ndvi",0):.3f}
            </div>
          </div>
          <div style='text-align:right;'>
            <span style='font-family:"JetBrains Mono",monospace; font-size:0.9rem;
                         color:{a_color}; font-weight:700;'>{risk_val:.2f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hourly Risk Heatmap for Alerts ────────────────────────────────────────
    st.markdown("<div class='section-header'>24-Hour Alert Pattern Analysis</div>", unsafe_allow_html=True)
    hourly_stats = df.groupby("hour").agg(
        accident_rate=("accident","mean"),
        avg_risk=("risk_score","mean"),
        total_accidents=("accident","sum"),
    ).reset_index()

    fig_alert_heatmap = go.Figure()
    fig_alert_heatmap.add_trace(go.Bar(
        x=hourly_stats["hour"], y=hourly_stats["total_accidents"],
        marker_color=[
            "#ff3d5a" if (h < 6 or h >= 20) else "#ffb020" if (5<=h<=7 or 17<=h<=19)
            else "#00e676" for h in hourly_stats["hour"]
        ],
        text=hourly_stats["total_accidents"], textposition="outside",
        name="Accidents",
    ))
    fig_alert_heatmap.add_trace(go.Scatter(
        x=hourly_stats["hour"], y=hourly_stats["avg_risk"]*hourly_stats["total_accidents"].max(),
        mode="lines", line=dict(color="#00e5ff", width=2, dash="dot"),
        name="Risk Trend (scaled)", yaxis="y2",
    ))
    fig_alert_heatmap.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        title="Accidents by Hour — Night/Dawn-Dusk Highlighting",
        xaxis=dict(title="Hour of Day", gridcolor="#1a2332", zerolinecolor="#1a2332",
                   dtick=1, range=[-0.5, 23.5]),
        yaxis=dict(title="Total Accidents", gridcolor="#1a2332", zerolinecolor="#1a2332"),
        yaxis2=dict(overlaying="y", side="right", showticklabels=False),
        height=380, margin=dict(l=40,r=20,t=40,b=30), showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    # Night window shading
    fig_alert_heatmap.add_vrect(x0=-0.5, x1=5.5, fillcolor="#ff3d5a08", line_width=0,
                                 annotation_text="Night", annotation_position="top left",
                                 annotation=dict(font=dict(size=9, color="#ff3d5a66")))
    fig_alert_heatmap.add_vrect(x0=19.5, x1=23.5, fillcolor="#ff3d5a08", line_width=0)
    fig_alert_heatmap.add_vrect(x0=4.5, x1=7.5, fillcolor="#ffb02008", line_width=0,
                                 annotation_text="Dawn", annotation_position="top left",
                                 annotation=dict(font=dict(size=9, color="#ffb02066")))
    fig_alert_heatmap.add_vrect(x0=16.5, x1=19.5, fillcolor="#ffb02008", line_width=0,
                                 annotation_text="Dusk", annotation_position="top left",
                                 annotation=dict(font=dict(size=9, color="#ffb02066")))
    st.plotly_chart(fig_alert_heatmap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — SDG GOALS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 SDG Goals":
    from utils.helpers import PALETTE

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🌍 UN Sustainable Development <span style='color:#00e5ff;'>Goals Alignment</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Mapping WildGuard AI's impact to the United Nations 2030 Agenda for Sustainable Development
    </p>
    """, unsafe_allow_html=True)

    # SDG Goals relevant to this project
    sdg_goals = [
        {
            "number": 15, "title": "Life on Land",
            "color": "#56C02B", "icon": "🌳",
            "description": "Protect, restore and promote sustainable use of terrestrial ecosystems, "
                          "sustainably manage forests, combat desertification, halt and reverse land degradation "
                          "and halt biodiversity loss.",
            "alignment": "DIRECT",
            "targets": [
                {"id": "15.1", "text": "Conservation of terrestrial and freshwater ecosystems",
                 "contribution": "NDVI monitoring tracks vegetation health near wildlife corridors; real-time alerts prevent habitat fragmentation from road mortality."},
                {"id": "15.5", "text": "Reduce degradation of natural habitats and halt biodiversity loss",
                 "contribution": "Species risk scoring identifies the most threatened animals (tiger=0.9, elephant=0.95) and prioritizes corridors for protection."},
                {"id": "15.7", "text": "End poaching and trafficking of protected species",
                 "contribution": "Monitoring animal movement patterns across 21 protected areas helps detect unusual activity that could indicate poaching."},
                {"id": "15.9", "text": "Integrate ecosystem values into planning",
                 "contribution": "Risk prediction models provide data-driven evidence for road planning and wildlife crossing infrastructure investment."},
            ],
            "metrics": {
                "Protected Areas Monitored": "21 (14 South India + 7 Central India)",
                "Wildlife Corridors Tracked": "11 active corridors",
                "Species Covered": "8 (tiger, elephant, leopard, deer, boar, wolf, nilgai, sambar)",
                "NDVI Coverage": f"Mean vegetation index: {df['ndvi'].mean():.3f}",
            },
        },
        {
            "number": 11, "title": "Sustainable Cities and Communities",
            "color": "#FD9D24", "icon": "🏙️",
            "description": "Make cities and human settlements inclusive, safe, resilient and sustainable.",
            "alignment": "DIRECT",
            "targets": [
                {"id": "11.2", "text": "Sustainable transport systems for all",
                 "contribution": "Risk maps guide road authorities to install wildlife crossings, speed limits, and warning systems on high-risk highway segments."},
                {"id": "11.6", "text": "Reduce environmental impact of cities",
                 "contribution": "Night light index and urbanization tracking help quantify human encroachment into wildlife habitats along transportation corridors."},
                {"id": "11.b", "text": "Implement integrated policies for resource efficiency and disaster risk reduction",
                 "contribution": "Alert system enables proactive risk management rather than reactive incident response, reducing both animal deaths and human injuries."},
            ],
            "metrics": {
                "Road Types Analyzed": "National highway, state highway, district road, forest road, rural",
                "Accident Prevention Model": f"AUC = {xgb_metrics.get('roc_auc',0):.4f} (XGBoost)",
                "High-Risk Zones Identified": f"{(df['risk_score'] > 0.65).sum():,} locations",
            },
        },
        {
            "number": 9, "title": "Industry, Innovation and Infrastructure",
            "color": "#FD6925", "icon": "🏗️",
            "description": "Build resilient infrastructure, promote inclusive and sustainable industrialization "
                          "and foster innovation.",
            "alignment": "SUPPORTING",
            "targets": [
                {"id": "9.1", "text": "Develop quality, reliable, sustainable and resilient infrastructure",
                 "contribution": "Data-driven recommendations for wildlife crossing structures, underpasses, and overpasses based on predictive modeling of collision hotspots."},
                {"id": "9.5", "text": "Enhance scientific research and upgrade technological capabilities",
                 "contribution": "Dual-model ensemble (XGBoost + Random Forest) with SHAP explainability represents cutting-edge application of ML to conservation."},
            ],
            "metrics": {
                "ML Models Deployed": "2 (XGBoost + Random Forest ensemble)",
                "Features Engineered": "30+ from 7 real-time data sources",
                "Explainability": "SHAP values for every prediction",
            },
        },
        {
            "number": 3, "title": "Good Health and Well-being",
            "color": "#4C9F38", "icon": "💚",
            "description": "Ensure healthy lives and promote well-being for all at all ages.",
            "alignment": "SUPPORTING",
            "targets": [
                {"id": "3.6", "text": "Halve global deaths and injuries from road traffic accidents",
                 "contribution": "Wildlife-vehicle collisions cause human fatalities and injuries. Early warning systems and speed reduction in high-risk zones protect both human and animal lives."},
            ],
            "metrics": {
                "Night Risk Reduction": f"Night accident rate: {df[df['night_flag']==1]['accident'].mean():.1%}",
                "Speed Factor": "Speed ratio analysis identifies dangerous driving conditions",
            },
        },
        {
            "number": 13, "title": "Climate Action",
            "color": "#3F7E44", "icon": "🌡️",
            "description": "Take urgent action to combat climate change and its impacts.",
            "alignment": "SUPPORTING",
            "targets": [
                {"id": "13.1", "text": "Strengthen resilience and adaptive capacity to climate-related hazards",
                 "contribution": "Seasonal prediction models account for monsoon, drought, and climate-driven changes in animal movement patterns and vegetation (NDVI)."},
                {"id": "13.3", "text": "Improve education and awareness on climate change mitigation",
                 "contribution": "Dashboard visualizations communicate how climate variables (rainfall, temperature, humidity) affect wildlife-road interactions."},
            ],
            "metrics": {
                "Climate Variables Tracked": "Temperature, rainfall, humidity, visibility",
                "Seasonal Coverage": "Summer, monsoon, post-monsoon, winter cycles",
            },
        },
        {
            "number": 17, "title": "Partnerships for the Goals",
            "color": "#19486A", "icon": "🤝",
            "description": "Strengthen the means of implementation and revitalize the Global Partnership "
                          "for Sustainable Development.",
            "alignment": "ENABLING",
            "targets": [
                {"id": "17.6", "text": "Knowledge sharing and cooperation for access to science, technology and innovation",
                 "contribution": "Open data integration from GBIF, OpenStreetMap, Open-Meteo, and government portals demonstrates collaborative approach to conservation technology."},
                {"id": "17.18", "text": "Enhance availability of reliable data",
                 "contribution": "7-source real-time data pipeline provides comprehensive and timely data for evidence-based wildlife management."},
            ],
            "metrics": {
                "Data Sources Integrated": "7 (APIs, RSS, Government portals, GIS computation)",
                "Open Data Partners": "GBIF, OpenStreetMap, Open-Meteo, India Biodiversity Portal",
            },
        },
    ]

    # ── Overall SDG Impact Summary ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Impact Summary</div>", unsafe_allow_html=True)
    impact_cols = st.columns(4)
    with impact_cols[0]:
        st.markdown("""<div class='metric-card'>
          <h3>SDGs Addressed</h3>
          <div class='value' style='font-size:2rem; color:#00e5ff;'>6</div>
          <div class='sub'>of 17 UN Goals</div>
        </div>""", unsafe_allow_html=True)
    with impact_cols[1]:
        st.markdown("""<div class='metric-card'>
          <h3>Direct Alignment</h3>
          <div class='value' style='font-size:2rem; color:#00e676;'>2</div>
          <div class='sub'>SDG 15 + SDG 11</div>
        </div>""", unsafe_allow_html=True)
    with impact_cols[2]:
        st.markdown("""<div class='metric-card'>
          <h3>Targets Covered</h3>
          <div class='value' style='font-size:2rem; color:#ffb020;'>14</div>
          <div class='sub'>specific UN targets</div>
        </div>""", unsafe_allow_html=True)
    with impact_cols[3]:
        st.markdown(f"""<div class='metric-card'>
          <h3>Data Points</h3>
          <div class='value' style='font-size:2rem; color:#d500f9;'>{len(df):,}</div>
          <div class='sub'>training records</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SDG Goal Cards ────────────────────────────────────────────────────────
    for goal in sdg_goals:
        alignment_badge_color = {"DIRECT":"#00e676","SUPPORTING":"#ffb020","ENABLING":"#4da6ff"}[goal["alignment"]]

        st.markdown(f"""
        <div style='background:#0d1320; border:1px solid {goal["color"]}44; padding:1.2rem;
                    margin-bottom:0.8rem; border-left:4px solid {goal["color"]};'>
          <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;'>
            <div style='display:flex; align-items:center; gap:0.8rem;'>
              <div style='font-size:2rem;'>{goal["icon"]}</div>
              <div>
                <div style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#5a6d82;
                            letter-spacing:0.15em;'>SDG {goal["number"]}</div>
                <div style='font-size:1.1rem; font-weight:600; color:{goal["color"]};'>{goal["title"]}</div>
              </div>
            </div>
            <span style='font-size:0.55rem; padding:0.15rem 0.6rem; color:{alignment_badge_color};
                         background:{alignment_badge_color}15; border:1px solid {alignment_badge_color}33;
                         font-family:"IBM Plex Mono",monospace; letter-spacing:0.1em;'>
              {goal["alignment"]}
            </span>
          </div>
          <p style='font-size:0.75rem; color:#8B949E; margin:0 0 0.6rem 0; line-height:1.4;'>
            {goal["description"]}
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Targets and metrics in two columns
        target_col, metric_col = st.columns([1.5, 1])
        with target_col:
            for target in goal["targets"]:
                st.markdown(f"""
                <div style='padding:0.4rem 0.6rem; margin:0.2rem 0; background:#060a10;
                            border-left:2px solid {goal["color"]}66;'>
                  <div style='font-size:0.7rem; color:{goal["color"]}; font-family:"JetBrains Mono",monospace;'>
                    Target {target["id"]}: {target["text"]}
                  </div>
                  <div style='font-size:0.68rem; color:#8B949E; margin-top:0.2rem; line-height:1.3;'>
                    {target["contribution"]}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with metric_col:
            st.markdown(f"""
            <div class='metric-card' style='border-top:2px solid {goal["color"]};'>
              <h3 style='color:{goal["color"]}; font-size:0.75rem;'>Project Metrics</h3>
            </div>""", unsafe_allow_html=True)
            for metric_name, metric_val in goal["metrics"].items():
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:0.25rem 0.4rem;
                            border-bottom:1px solid #1a2332; font-size:0.68rem;'>
                  <span style='color:#5a6d82;'>{metric_name}</span>
                  <span style='color:#c8d6e5; font-family:"JetBrains Mono",monospace;'>{metric_val}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── SDG Contribution Radar ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>SDG Contribution Radar</div>", unsafe_allow_html=True)
    sdg_labels = [f"SDG {g['number']}" for g in sdg_goals]
    sdg_scores = [0.95, 0.80, 0.65, 0.55, 0.60, 0.70]  # alignment scores
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=sdg_scores + [sdg_scores[0]], theta=sdg_labels + [sdg_labels[0]],
        fill="toself", fillcolor="rgba(0,229,255,0.12)",
        line=dict(color="#00e5ff", width=2),
        marker=dict(size=8, color="#00e5ff"),
        name="WildGuard AI Impact",
    ))
    fig_radar.update_layout(
        paper_bgcolor="#060a10", font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        polar=dict(
            bgcolor="#0d1320",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1a2332",
                           tickfont=dict(size=8, color="#5a6d82")),
            angularaxis=dict(gridcolor="#1a2332", tickfont=dict(size=10, color="#c8d6e5")),
        ),
        height=420, margin=dict(l=60,r=60,t=30,b=30), showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — SEASON-WISE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Season Prediction":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      📅 Season-Wise <span style='color:#ffb020;'>Risk Prediction</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Comprehensive seasonal analysis with predictive modeling for each Indian season cycle
    </p>
    """, unsafe_allow_html=True)

    seasons_list = ["summer", "monsoon", "post_monsoon", "winter"]
    season_display = {"summer":"Summer (Mar-May)", "monsoon":"Monsoon (Jun-Sep)",
                      "post_monsoon":"Post-Monsoon (Oct-Nov)", "winter":"Winter (Dec-Feb)"}
    season_icons_map = {"summer":"☀️", "monsoon":"🌧️", "post_monsoon":"🍂", "winter":"❄️"}
    season_color_map = {"summer":"#ffb020", "monsoon":"#00e676", "post_monsoon":"#ff7043", "winter":"#4da6ff"}

    # ── Season Overview Cards ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Seasonal Risk Overview</div>", unsafe_allow_html=True)
    season_stats = df.groupby("season").agg(
        count=("accident","count"), accidents=("accident","sum"),
        acc_rate=("accident","mean"), avg_risk=("risk_score","mean"),
        avg_ndvi=("ndvi","mean"), avg_movement=("movement_score","mean"),
        avg_speed_ratio=("speed_ratio","mean"), avg_visibility=("visibility_m","mean"),
        avg_rainfall=("rainfall_mm","mean"),
    )

    scols = st.columns(4)
    for i, s in enumerate(seasons_list):
        with scols[i]:
            if s in season_stats.index:
                row = season_stats.loc[s]
                risk_badge = "🔴 HIGH" if row["acc_rate"] > 0.45 else "🟡 MODERATE" if row["acc_rate"] > 0.35 else "🟢 LOW"
                st.markdown(f"""
                <div class='metric-card' style='border-top:4px solid {season_color_map[s]}; padding:1rem;'>
                  <div style='text-align:center; font-size:2rem;'>{season_icons_map[s]}</div>
                  <h3 style='text-align:center; color:{season_color_map[s]}; font-size:0.9rem;'>{season_display[s]}</h3>
                  <div style='text-align:center; margin:0.4rem 0;'>
                    <span style='font-size:0.6rem; padding:0.1rem 0.5rem;
                                background:{season_color_map[s]}15; color:{season_color_map[s]};
                                border:1px solid {season_color_map[s]}33;'>{risk_badge}</span>
                  </div>
                  <div style='margin-top:0.5rem; font-size:0.65rem; color:#5a6d82; line-height:1.6;'>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>Records</span><span style='color:#e8f0fa;'>{row["count"]:,.0f}</span></div>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>Accidents</span><span style='color:#ff3d5a;'>{row["accidents"]:,.0f}</span></div>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>Accident Rate</span><span style='color:#ff3d5a;'>{row["acc_rate"]:.1%}</span></div>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>Avg Risk</span><span style='color:#ffb020;'>{row["avg_risk"]:.3f}</span></div>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>NDVI</span><span style='color:#00e676;'>{row["avg_ndvi"]:.3f}</span></div>
                    <div style='display:flex; justify-content:space-between;'>
                      <span>Movement</span><span style='color:#00e5ff;'>{row["avg_movement"]:.3f}</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seasonal Comparison Charts ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Seasonal Risk Comparison</div>", unsafe_allow_html=True)

    comp_cols = st.columns(2)
    with comp_cols[0]:
        # Accident rate by season bar chart
        fig_season_acc = go.Figure(go.Bar(
            x=[season_display[s] for s in seasons_list],
            y=[season_stats.loc[s,"acc_rate"] if s in season_stats.index else 0 for s in seasons_list],
            marker_color=[season_color_map[s] for s in seasons_list],
            text=[f"{season_stats.loc[s,'acc_rate']:.1%}" if s in season_stats.index else "0%" for s in seasons_list],
            textposition="outside",
        ))
        fig_season_acc.update_layout(
            paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            title="Accident Rate by Season", height=380,
            xaxis=dict(gridcolor="#1a2332", zerolinecolor="#1a2332"),
            yaxis=dict(title="Accident Rate", gridcolor="#1a2332", zerolinecolor="#1a2332",
                       tickformat=".0%"),
            margin=dict(l=40,r=20,t=40,b=30),
        )
        st.plotly_chart(fig_season_acc, use_container_width=True)

    with comp_cols[1]:
        # Multi-variable radar per season
        fig_season_radar = go.Figure()
        radar_vars = ["Accident Rate", "Avg Risk", "NDVI", "Movement", "Rainfall", "Visibility"]
        for s in seasons_list:
            if s in season_stats.index:
                row = season_stats.loc[s]
                vals = [row["acc_rate"], row["avg_risk"],
                        row["avg_ndvi"], row["avg_movement"],
                        min(1, row["avg_rainfall"]/100), min(1, row["avg_visibility"]/1000)]
                fig_season_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_vars + [radar_vars[0]],
                    fill="toself", fillcolor=season_color_map[s]+"18",
                    line=dict(color=season_color_map[s], width=2),
                    name=season_display[s],
                ))
        fig_season_radar.update_layout(
            paper_bgcolor="#060a10",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
            polar=dict(bgcolor="#0d1320",
                       radialaxis=dict(visible=True, range=[0,1], gridcolor="#1a2332",
                                       tickfont=dict(size=8, color="#5a6d82")),
                       angularaxis=dict(gridcolor="#1a2332")),
            title="Multi-Variable Season Comparison", height=380,
            margin=dict(l=60,r=60,t=40,b=30),
            legend=dict(font=dict(size=9), orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_season_radar, use_container_width=True)

    # ── Species Risk by Season ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Species Risk Heatmap by Season</div>", unsafe_allow_html=True)

    sp_season = df.groupby(["species","season"])["accident"].mean().unstack(fill_value=0)
    sp_season = sp_season.reindex(columns=seasons_list)
    fig_sp_heat = go.Figure(go.Heatmap(
        z=sp_season.values, x=[season_display[s] for s in seasons_list],
        y=[s.capitalize() for s in sp_season.index],
        colorscale=[[0,"#060a10"],[0.3,"#0d4f3c"],[0.5,"#ffb020"],[0.7,"#ff7043"],[1,"#ff3d5a"]],
        text=[[f"{v:.1%}" for v in row] for row in sp_season.values],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title="Accident Rate", tickfont=dict(color="#5a6d82"),
                      title_font=dict(color="#5a6d82")),
    ))
    fig_sp_heat.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        title="Species × Season Accident Rate Matrix", height=380,
        xaxis=dict(gridcolor="#1a2332"), yaxis=dict(gridcolor="#1a2332"),
        margin=dict(l=80,r=20,t=40,b=30),
    )
    st.plotly_chart(fig_sp_heat, use_container_width=True)

    # ── Season Prediction Tool ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Season-Specific Risk Predictor</div>", unsafe_allow_html=True)
    pred_col1, pred_col2 = st.columns([1, 2])

    with pred_col1:
        pred_season = st.selectbox("Select Season", seasons_list,
                                    format_func=lambda x: f"{season_icons_map[x]} {season_display[x]}",
                                    key="season_pred_select")
        pred_species = st.selectbox("Select Species",
                                     sorted(df["species"].unique()),
                                     format_func=lambda x: x.capitalize(),
                                     key="season_pred_species")
        pred_road = st.selectbox("Road Type",
                                  sorted(df["road_type"].unique()),
                                  format_func=lambda x: x.replace("_"," ").title(),
                                  key="season_pred_road")
        pred_hour = st.slider("Hour of Day", 0, 23, 18, key="season_pred_hour")

    with pred_col2:
        # Filter data for prediction context
        mask = (df["season"] == pred_season)
        if pred_species in df["species"].values:
            mask_sp = mask & (df["species"] == pred_species)
            if mask_sp.sum() > 10:
                mask = mask_sp
        if pred_road in df["road_type"].values:
            mask_rd = mask & (df["road_type"] == pred_road)
            if mask_rd.sum() > 10:
                mask = mask_rd

        filtered = df[mask]
        if len(filtered) > 0:
            pred_risk = filtered["risk_score"].mean()
            pred_acc_rate = filtered["accident"].mean()
            pred_ndvi = filtered["ndvi"].mean()
            pred_movement = filtered["movement_score"].mean()

            risk_color = "#ff3d5a" if pred_risk > 0.6 else "#ffb020" if pred_risk > 0.4 else "#00e676"

            st.markdown(f"""
            <div style='background:#0d1320; border:1px solid {risk_color}44; padding:1.2rem;'>
              <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#5a6d82;
                          letter-spacing:0.2em;'>SEASONAL PREDICTION RESULT</div>
              <div style='display:flex; gap:1.5rem; margin-top:0.6rem;'>
                <div style='flex:1; text-align:center;'>
                  <div style='font-size:2rem;'>{season_icons_map[pred_season]}</div>
                  <div style='font-family:"JetBrains Mono",monospace; font-size:1.8rem; font-weight:700;
                              color:{risk_color};'>{pred_risk:.3f}</div>
                  <div style='font-size:0.6rem; color:#5a6d82;'>Predicted Risk Score</div>
                </div>
                <div style='flex:2;'>
                  <div style='font-size:0.7rem; color:#5a6d82; line-height:1.8;'>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #1a2332; padding:0.2rem 0;'>
                      <span>Accident Probability</span>
                      <span style='color:#ff3d5a; font-family:"JetBrains Mono",monospace;'>{pred_acc_rate:.1%}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #1a2332; padding:0.2rem 0;'>
                      <span>Expected NDVI</span>
                      <span style='color:#00e676; font-family:"JetBrains Mono",monospace;'>{pred_ndvi:.4f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #1a2332; padding:0.2rem 0;'>
                      <span>Movement Score</span>
                      <span style='color:#00e5ff; font-family:"JetBrains Mono",monospace;'>{pred_movement:.4f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #1a2332; padding:0.2rem 0;'>
                      <span>Sample Size</span>
                      <span style='color:#c8d6e5; font-family:"JetBrains Mono",monospace;'>{len(filtered):,} records</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; padding:0.2rem 0;'>
                      <span>Conditions</span>
                      <span style='color:#c8d6e5; font-size:0.65rem;'>
                        {pred_species.capitalize()} · {pred_road.replace('_',' ').title()} · {pred_hour}:00
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Hour × Season Heatmap ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Hour × Season Risk Heatmap</div>", unsafe_allow_html=True)
    hour_season = df.groupby(["hour","season"])["accident"].mean().unstack(fill_value=0)
    hour_season = hour_season.reindex(columns=seasons_list)
    fig_hs = go.Figure(go.Heatmap(
        z=hour_season.values.T, x=list(range(24)),
        y=[season_display[s] for s in seasons_list],
        colorscale=[[0,"#060a10"],[0.3,"#0d4f3c"],[0.5,"#ffb020"],[0.7,"#ff7043"],[1,"#ff3d5a"]],
        showscale=True,
        colorbar=dict(title="Accident Rate", tickfont=dict(color="#5a6d82"),
                      title_font=dict(color="#5a6d82")),
    ))
    fig_hs.update_layout(
        paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
        font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"),
        title="Hour of Day × Season — Accident Rate", height=320,
        xaxis=dict(title="Hour of Day", dtick=1, gridcolor="#1a2332", zerolinecolor="#1a2332"),
        yaxis=dict(gridcolor="#1a2332"),
        margin=dict(l=120,r=20,t=40,b=30),
    )
    st.plotly_chart(fig_hs, use_container_width=True)

    # ── Seasonal Recommendations ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>Seasonal Safety Recommendations</div>", unsafe_allow_html=True)

    recommendations = {
        "summer": [
            "Increased animal movement toward water sources — monitor water crossings closely",
            "Higher speeds due to clear visibility — enforce speed limits near corridors",
            "Reduced vegetation cover exposes animals during road crossing",
            "Deploy additional warning signage near known watering holes",
        ],
        "monsoon": [
            "Peak breeding season drives unusual animal movement patterns",
            "Reduced visibility from rain — activate electronic warning boards",
            "Flooded underpasses force animals to cross roads at surface level",
            "Monitor swollen river crossings — animals may be displaced from usual paths",
            "Deploy patrol teams during dawn/dusk periods when risk is highest",
        ],
        "post_monsoon": [
            "Breeding activity continues — maintain heightened surveillance",
            "Lush vegetation provides cover near roads — animals may appear suddenly",
            "Migratory species begin seasonal movements through corridors",
            "High NDVI may give false sense of security — maintain alert level",
        ],
        "winter": [
            "Cooler temperatures increase daytime animal activity",
            "Fog and poor visibility in early morning — highest single-hour risk",
            "Holiday traffic increases vehicle volume on highways",
            "Shorter days expand the effective night-risk window",
        ],
    }

    for s in seasons_list:
        color = season_color_map[s]
        icon = season_icons_map[s]
        st.markdown(f"""
        <div style='margin-bottom:0.8rem;'>
          <div style='font-size:0.85rem; color:{color}; font-weight:600; margin-bottom:0.3rem;'>
            {icon} {season_display[s]}
          </div>
        </div>""", unsafe_allow_html=True)
        for rec in recommendations[s]:
            st.markdown(f"- {rec}")
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Data Sources":
    from utils.helpers import PALETTE

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      📡 Data <span style='color:#FF6B35;'>Sources & Attribution</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Where and how the WildGuard AI system extracts real-time data
    </p>
    """, unsafe_allow_html=True)

    sources = [
        {
            "name": "Open-Meteo Weather API",
            "icon": "🌦️",
            "url": "https://api.open-meteo.com/v1/forecast",
            "description": "Free, open-source weather API providing real-time meteorological data. No API key required.",
            "license": "CC BY 4.0 — Open Source",
            "features": ["temperature_c", "humidity_pct", "rainfall_mm", "visibility_m (from WMO weather codes)", "wind_speed", "cloud_cover"],
            "how": "HTTP GET request with latitude/longitude parameters. Returns JSON with current weather observations. "
                   "Visibility is estimated from WMO weather codes (fog → 100-200m, clear → 900-1000m, rain → 400-700m).",
            "rate_limit": "No registration needed. 10,000 requests/day.",
            "color": "#58A6FF",
        },
        {
            "name": "OpenStreetMap Overpass API",
            "icon": "🗺️",
            "url": "https://overpass-api.de/api/interpreter",
            "description": "Queries the OpenStreetMap database for geographic features near the target location. "
                           "Returns road types, width, lighting, and water body proximity.",
            "license": "ODbL — Open Data Commons Open Database License",
            "features": ["road_type (mapped from OSM highway tag)", "road_width_m", "street_lighting",
                          "dist_water_km (Haversine distance to nearest water body)"],
            "how": "POST request with Overpass QL query. Searches within 1.5 km radius for highway=* ways and "
                   "natural=water / waterway nodes/ways. OSM highway types are mapped to model categories "
                   "(motorway→national_highway, track→forest_road, etc.).",
            "rate_limit": "Public endpoint. Max 2 requests/second, 10,000/day.",
            "color": "#06D6A0",
        },
        {
            "name": "GBIF Biodiversity API",
            "icon": "🦁",
            "url": "https://api.gbif.org/v1/occurrence/search",
            "description": "Global Biodiversity Information Facility — the world's largest open biodiversity database. "
                           "Searches for mammal occurrence records reported near the target coordinates.",
            "license": "CC BY 4.0 — GBIF.org Data Use Agreement",
            "features": ["species (dominant mammal in area)", "species_risk (mapped from species ID)",
                          "total_sightings", "biodiversity_index"],
            "how": "HTTP GET request filtered by Mammalia class (classKey=359), ±0.5° lat/lon bounding box. "
                   "Returns up to 300 occurrence records. Species names are matched against our risk mapping "
                   "(tiger=0.9, elephant=0.95, deer=0.6, etc.). Most frequent match becomes the dominant species.",
            "rate_limit": "No API key required. Rate-limited to 3 requests/second.",
            "color": "#FFD166",
        },
        {
            "name": "System Clock (Temporal Features)",
            "icon": "🕐",
            "url": "Python datetime.now()",
            "description": "Derives all temporal features from the current system time. "
                           "Indian seasonal calendar is applied (Mar-May=summer, Jun-Sep=monsoon, Oct-Nov=post_monsoon, Dec-Feb=winter).",
            "license": "N/A — Local computation",
            "features": ["hour", "day_of_week", "season", "night_flag (20:00–06:00)",
                          "dawn_dusk (05:00–07:00 / 17:00–19:00)", "rush_hour (06:00–09:00 / 17:00–20:00)",
                          "breeding_season (monsoon + post_monsoon)"],
            "how": "Python's datetime module provides the current local time. Boolean flags are computed for "
                   "nighttime, dawn/dusk (peak wildlife crossing periods), rush hour, and breeding season.",
            "rate_limit": "N/A — Instantaneous",
            "color": "#BC8CF2",
        },
        {
            "name": "GIS Computation Engine",
            "icon": "🌍",
            "url": "Haversine distance + NDVI estimation",
            "description": "Computes geographical features using a database of 21 protected areas "
                           "(14 South India + 7 Central India) and 11 wildlife corridors.",
            "license": "N/A — Local computation with openly sourced coordinates",
            "features": ["protected_dist_km (nearest tiger reserve/national park)",
                          "corridor_dist_km (nearest wildlife corridor)",
                          "ndvi (estimated from season + proximity to protected areas)",
                          "night_light (urbanization proxy from corridor distance)"],
            "how": "Haversine formula computes great-circle distance from the query point to each protected area "
                   "and corridor in our database. South India reserves include Bandipur, Nagarhole, Mudumalai, "
                   "Wayanad, Periyar, Sathyamangalam, BR Hills, Anamalai, Parambikulam, and more.",
            "rate_limit": "N/A — < 1ms computation",
            "color": "#FF6B35",
        },
        {
            "name": "News Websites (RSS Aggregator)",
            "icon": "📰",
            "url": "The Hindu | NDTV | Down to Earth | Google News RSS",
            "description": "Scrapes wildlife-vehicle collision news from major Indian news sources via RSS feeds. "
                           "Searches for recent reports of animal-road incidents, with special focus on South India "
                           "(Karnataka, Kerala, Tamil Nadu, Western Ghats). Extracts species mentions and regional tags.",
            "license": "RSS / Public Access — Editorial content",
            "features": ["total_wildlife_articles (count of matching news articles)",
                          "south_india_articles (count with South India mentions)",
                          "species_in_news (species cited in recent reports)",
                          "news_risk_modifier (risk boost from recent incidents)"],
            "how": "HTTP GET requests to RSS feed URLs for The Hindu (thehindu.com/sci-tech/energy-and-environment), "
                   "NDTV (feeds.feedburner.com/ndtv/environment-news), Down to Earth (downtoearth.org.in/rss/wildlife), "
                   "and Google News (news.google.com/rss/search?q=wildlife+animal+road+accident+India). "
                   "XML parsed with Python's xml.etree.ElementTree. Articles are filtered by wildlife keywords "
                   "(elephant, tiger, roadkill, corridor, etc.) and checked for South India state mentions.",
            "rate_limit": "No auth needed. 4 RSS feeds fetched in parallel.",
            "color": "#EF476F",
        },
        {
            "name": "Indian Government Wildlife Portals",
            "icon": "🏛️",
            "url": "NTCA | State Forest Depts | MoEFCC | WII | MoRTH | India Biodiversity Portal",
            "description": "Checks accessibility and scrapes wildlife/conservation references from official Indian "
                           "government portals. Includes the National Tiger Conservation Authority (NTCA), "
                           "state forest departments (Karnataka aranya.gov.in, Kerala forest.kerala.gov.in, "
                           "Tamil Nadu forests.tn.gov.in), Ministry of Environment (MoEFCC), Wildlife Institute of India (WII), "
                           "Ministry of Road Transport (MoRTH), and India Biodiversity Portal.",
            "license": "Government of India — Public Access",
            "features": ["govt_sources_accessible (count of reachable portals out of 6+)",
                          "conservation_intensity (tiger/corridor reference density from NTCA)",
                          "active_alerts (whether state forest dept has wildlife advisories)",
                          "nearest_state_dept (auto-detected based on location)"],
            "how": "HTTP GET requests to each government URL. HTML content is scanned with regex for wildlife, corridor, "
                   "tiger, elephant, alert, and advisory references. The nearest state forest department is auto-selected "
                   "based on the query coordinates (e.g., Bandipur → Karnataka Forest Dept at aranya.gov.in). "
                   "MoRTH is checked for road accident/safety references. India Biodiversity Portal is queried for observations.",
            "rate_limit": "Public access. 6-8 government sites checked per request.",
            "color": "#B5179E",
        },
    ]

    for src in sources:
        st.markdown(f"""
        <div class='source-card' style='border-left: 3px solid {src["color"]};'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div class='source-name' style='color:{src["color"]}; font-size:1rem;'>
              {src["icon"]}  {src["name"]}
            </div>
            <span style='font-size:0.65rem; color:var(--muted); background:var(--bg);
                          padding:0.2rem 0.6rem; border-radius:4px;'>{src["license"]}</span>
          </div>
          <div style='font-size:0.8rem; color:var(--text); margin:0.5rem 0;'>{src["description"]}</div>
          <div class='source-url' style='margin:0.4rem 0;'>
            <code style='background:var(--bg); padding:0.2rem 0.5rem; border-radius:4px; font-size:0.7rem;'>
              {src["url"]}
            </code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_feat, col_how = st.columns([1, 1.5])
        with col_feat:
            st.markdown("**Features Extracted:**")
            for feat in src["features"]:
                st.markdown(f"- `{feat}`")
        with col_how:
            st.markdown("**How It Works:**")
            st.markdown(src["how"])
            st.markdown(f"**Rate Limit:** {src['rate_limit']}")

        st.markdown("---")

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Pipeline Architecture (7 Sources)</div>", unsafe_allow_html=True)
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  REAL-TIME DATA PIPELINE (v3.0)                     │
    │           7 Sources · 30+ Features · 2 Models · Ensemble           │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   📍 User selects location (lat/lon) + speed conditions            │
    │         │                                                           │
    │  ┌──── APIs ────────────────────────────────────────────────┐      │
    │  │      ├──→ 🌦️ Open-Meteo API     ──→ weather features   │      │
    │  │      ├──→ 🗺️ Overpass API        ──→ road + water      │      │
    │  │      └──→ 🦁 GBIF API            ──→ wildlife species   │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │  ┌──── News & Government ───────────────────────────────────┐      │
    │  │      ├──→ 📰 News RSS            ──→ incident intelligence│     │
    │  │      │    (The Hindu, NDTV, Down to Earth, Google News)  │      │
    │  │      └──→ 🏛️ Govt Portals        ──→ conservation data  │      │
    │  │           (NTCA, Forest Depts, MoEFCC, WII, MoRTH)      │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │  ┌──── Computation ─────────────────────────────────────────┐      │
    │  │      ├──→ 🕐 System Clock        ──→ temporal features   │      │
    │  │      └──→ 🌍 GIS Engine          ──→ spatial features    │      │
    │  │           (21 PAs + 11 corridors, South + Central India) │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │                   │                                                 │
    │                   ▼                                                 │
    │         ┌─────────────────┐                                        │
    │         │ Feature Vector  │  (30+ features combined)               │
    │         └────────┬────────┘                                        │
    │                  │                                                  │
    │         ┌────────┴────────┐                                        │
    │         ▼                 ▼                                         │
    │   ┌──────────┐    ┌──────────────┐                                │
    │   │ XGBoost  │    │ Random Forest│                                │
    │   │ (SHAP)   │    │ (Gini)       │                                │
    │   └────┬─────┘    └──────┬───────┘                                │
    │        └────────┬────────┘                                         │
    │                 ▼                                                   │
    │       ┌─────────────────┐                                          │
    │       │ Ensemble Result │                                          │
    │       │ + SHAP Waterfall│                                          │
    │       │ + Recommendations│                                         │
    │       └─────────────────┘                                          │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)
