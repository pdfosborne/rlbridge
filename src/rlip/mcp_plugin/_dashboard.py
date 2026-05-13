"""
Live Training Dashboard
========================
A lightweight localhost web server that displays RL training progress
in real time.  No external dependencies — uses Python's built-in
``http.server``.

The dashboard page auto-refreshes every 2 seconds and shows:

* Per-agent reward curve (SVG sparkline)
* Current episode, epsilon, best reward so far
* Latest policy snapshot (text table or board diagram)
* Environment and agent metadata

The server runs in a **daemon thread** so it shuts down automatically when
the Python process exits.

Usage (internal)
----------------
    from rlip.mcp_plugin._dashboard import dashboard, start_dashboard
    url = start_dashboard()           # idempotent — returns URL if already running
    dashboard.update(agent_id, ...)   # called by _ProgressEnv each episode
"""

from __future__ import annotations

import html
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

# ── Thread-safe state store ───────────────────────────────────────────────────

class _AgentState:
    """Mutable state for one training run."""

    __slots__ = (
        "agent_id", "agent_type", "env_id",
        "use_language_state", "uses_instructions", "instructions",
        "n_episodes", "completed", "episode_rewards",
        "epsilon", "best_reward", "last_reward",
        "policy_text", "policy_gif_b64", "policy_frames",
        "started_at", "updated_at",
    )

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        env_id: str,
        n_episodes: int,
        use_language_state: bool = False,
        uses_instructions: bool = False,
        instructions: Optional[list[str]] = None,
    ) -> None:
        self.agent_id        = agent_id
        self.agent_type      = agent_type
        self.env_id          = env_id
        self.use_language_state = use_language_state
        self.uses_instructions = uses_instructions
        self.instructions = list(instructions or [])
        self.n_episodes      = n_episodes
        self.completed       = 0
        self.episode_rewards: list[float] = []
        self.epsilon         = 1.0
        self.best_reward     = float("-inf")
        self.last_reward     = 0.0
        self.policy_text     = ""          # formatted text snapshot of current policy
        self.policy_gif_b64  = ""          # base64 GIF of the best episode
        self.policy_frames: list[str] = [] # ANSI text frames (fallback)
        self.started_at      = time.time()
        self.updated_at      = time.time()


class TrainingDashboard:
    """
    Thread-safe registry of active / completed training runs.

    Call :meth:`register` when a run starts, :meth:`update` each episode,
    and optionally :meth:`finish` when done.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._agents: dict[str, _AgentState] = {}  # ordered by insertion
        self._version = 0  # increments on every update (for SSE / polling)

    def register(
        self,
        agent_id: str,
        agent_type: str,
        env_id: str,
        n_episodes: int,
        use_language_state: bool = False,
        uses_instructions: bool = False,
        instructions: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            self._agents[agent_id] = _AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                env_id=env_id,
                n_episodes=n_episodes,
                use_language_state=use_language_state,
                uses_instructions=uses_instructions,
                instructions=instructions,
            )
            self._version += 1

    def update(
        self,
        agent_id: str,
        completed: int,
        last_reward: float,
        epsilon: float,
        policy_text: str = "",
    ) -> None:
        with self._lock:
            state = self._agents.get(agent_id)
            if state is None:
                return
            state.completed   = completed
            state.last_reward = last_reward
            state.epsilon     = epsilon
            state.episode_rewards.append(last_reward)
            if last_reward > state.best_reward:
                state.best_reward = last_reward
            if policy_text:
                state.policy_text = policy_text
            state.updated_at  = time.time()
            self._version += 1

    def finish(
        self,
        agent_id: str,
        policy_text: str = "",
        policy_gif_b64: str = "",
        policy_frames: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            state = self._agents.get(agent_id)
            if state is None:
                return
            if policy_text:
                state.policy_text = policy_text
            if policy_gif_b64:
                state.policy_gif_b64 = policy_gif_b64
            if policy_frames:
                state.policy_frames = policy_frames
            # Mark as fully complete; update() may have already set this.
            state.completed  = state.n_episodes
            state.updated_at = time.time()
            self._version += 1

    def snapshot(self) -> tuple[int, list[_AgentState]]:
        """Return (version, list of states) under a single lock."""
        with self._lock:
            return self._version, list(self._agents.values())

    def get_version(self) -> int:
        with self._lock:
            return self._version


# Global singleton — imported by _tools_agents and the HTTP handler
dashboard = TrainingDashboard()


# ── SVG sparkline helpers ─────────────────────────────────────────────────────

def _sparkline_svg(rewards: list[float], width: int = 260, height: int = 50) -> str:
    """Render a list of reward values as a small inline SVG polyline."""
    if len(rewards) < 2:
        return f'<svg width="{width}" height="{height}"><text x="4" y="20" font-size="11" fill="#888">collecting data…</text></svg>'

    mn = min(rewards)
    mx = max(rewards)
    span = mx - mn if mx != mn else 1.0

    def px(i: int, v: float) -> tuple[float, float]:
        x = 4 + (i / (len(rewards) - 1)) * (width - 8)
        y = height - 4 - ((v - mn) / span) * (height - 8)
        return x, y

    points  = " ".join(f"{px(i,v)[0]:.1f},{px(i,v)[1]:.1f}" for i, v in enumerate(rewards))

    # Smoothed rolling mean overlay (window = 10 %)
    w = max(1, len(rewards) // 10)
    means: list[float] = []
    for i in range(len(rewards)):
        window = rewards[max(0, i - w): i + 1]
        means.append(sum(window) / len(window))
    mean_pts = " ".join(f"{px(i,v)[0]:.1f},{px(i,v)[1]:.1f}" for i, v in enumerate(means))

    # Filled area under the mean line
    first_x, first_y = px(0, means[0])
    last_x,  last_y  = px(len(means) - 1, means[-1])
    area_pts = (
        f"{first_x:.1f},{height - 4} "
        + mean_pts
        + f" {last_x:.1f},{height - 4}"
    )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polygon points="{area_pts}" fill="rgba(59,130,246,0.15)"/>'
        f'<polyline points="{points}" fill="none" stroke="rgba(59,130,246,0.35)" stroke-width="1"/>'
        f'<polyline points="{mean_pts}" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
        f'<text x="4" y="10" font-size="9" fill="#64748b">{mx:.2f}</text>'
        f'<text x="4" y="{height - 5}" font-size="9" fill="#64748b">{mn:.2f}</text>'
        f'</svg>'
    )


def _progress_bar(completed: int, total: int, width: int = 200) -> str:
    pct = min(1.0, completed / max(total, 1))
    filled = int(pct * width)
    color  = "#22c55e" if pct >= 1.0 else "#3b82f6"
    pct_txt = f"{pct * 100:.1f} %"
    return (
        f'<div style="background:#e2e8f0;border-radius:4px;width:{width}px;height:10px;overflow:hidden;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{filled}px;height:100%"></div>'
        f'</div> <span style="font-size:12px;color:#475569">{completed}/{total} ({pct_txt})</span>'
    )


# ── HTML page builder ─────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: ui-sans-serif, system-ui, sans-serif; background: #0f172a;
       color: #e2e8f0; min-height: 100vh; padding: 24px; }
h1   { font-size: 1.4rem; font-weight: 700; color: #f8fafc;
       margin-bottom: 4px; letter-spacing: -0.02em; }
.subtitle { font-size: 0.8rem; color: #64748b; margin-bottom: 24px; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 10px;
        padding: 18px 20px; margin-bottom: 16px; }
.card-header { display: flex; align-items: baseline; gap: 10px;
               margin-bottom: 12px; }
.env-badge   { font-size: 0.7rem; background: #0ea5e9; color: #fff;
               border-radius: 4px; padding: 2px 7px; font-weight: 600; }
.type-badge  { font-size: 0.7rem; background: #7c3aed; color: #fff;
               border-radius: 4px; padding: 2px 7px; }
.meta-badge  { font-size: 0.68rem; background: #334155; color: #e2e8f0;
               border-radius: 4px; padding: 2px 7px; border: 1px solid #475569; }
.done-badge  { font-size: 0.7rem; background: #22c55e; color: #fff;
               border-radius: 4px; padding: 2px 7px; }
.metrics     { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 12px; }
.metric      { display: flex; flex-direction: column; }
.metric-val  { font-size: 1.2rem; font-weight: 700; color: #f8fafc; }
.metric-lbl  { font-size: 0.7rem; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.05em; }
.chart-wrap  { margin: 8px 0; }
.policy-pre  { font-family: ui-monospace, monospace; font-size: 0.72rem;
               white-space: pre; background: #0f172a; border-radius: 6px;
               padding: 10px 12px; overflow-x: auto; color: #94a3b8;
               max-height: 260px; overflow-y: auto;
               border: 1px solid #1e3a5f; margin-top: 10px; }
.policy-lbl  { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.06em; margin-top: 10px; margin-bottom: 4px; }
.policy-grid { display: grid; grid-template-columns: minmax(180px, 1fr) 2fr;
               gap: 10px; align-items: start; }
.inst-box    { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 6px;
               padding: 8px; font-size: 0.72rem; color: #cbd5e1; }
.inst-list   { margin: 0; padding-left: 16px; }
.inst-list li { margin-bottom: 5px; line-height: 1.25; overflow-wrap: anywhere; }
.idle        { color: #475569; font-size: 0.9rem; padding: 24px 0; text-align: center; }
#refresh-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
               background: #22c55e; margin-left: 6px; vertical-align: middle; }
#ts          { font-size: 0.7rem; color: #475569; }
"""

_JS = """
let _ver = -1;
async function poll() {
  try {
    const r = await fetch('/version');
    const v = await r.json();
    if (v.version !== _ver) {
      _ver = v.version;
      const page = await fetch('/');
      const txt  = await page.text();
      const parser = new DOMParser();
      const doc    = parser.parseFromString(txt, 'text/html');
      document.getElementById('main').innerHTML =
          doc.getElementById('main').innerHTML;
      document.getElementById('ts').textContent =
          'Updated ' + new Date().toLocaleTimeString();
      // Flash refresh dot
      const dot = document.getElementById('refresh-dot');
      if (dot) {
        dot.style.background = '#f59e0b';
        setTimeout(() => { dot.style.background = '#22c55e'; }, 300);
      }
    }
  } catch(e) {}
  setTimeout(poll, 1500);
}
poll();
"""


def _render_agent_card(state: _AgentState) -> str:
    done    = state.completed >= state.n_episodes
    elapsed = time.time() - state.started_at
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    elapsed_str = (f"{h}h " if h else "") + f"{m:02d}m {s:02d}s"

    # Rolling mean of last 50 episodes
    recent       = state.episode_rewards[-50:]
    mean_recent  = sum(recent) / len(recent) if recent else 0.0

    done_badge = '<span class="done-badge">✓ Done</span>' if done else ""
    lang_tag = '<span class="meta-badge">Language translation</span>' if state.use_language_state else ''
    instr_tag = '<span class="meta-badge">Instructions</span>' if state.uses_instructions else ''

    instruction_panel = ""
    if state.uses_instructions:
        items = state.instructions or ["(instruction text unavailable)"]
        li = "".join(f"<li>{html.escape(it)}</li>" for it in items)
        instruction_panel = (
            '<div class="inst-box">'
            '<div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Instructions used</div>'
            f'<ul class="inst-list">{li}</ul>'
            '</div>'
        )

    # ── Policy render section ─────────────────────────────────────────────────
    policy_section = ""
    if done:
        if state.policy_gif_b64:
            # Animated GIF of the best training episode
            if instruction_panel:
                policy_section = (
                    '<div class="policy-lbl">Optimal policy replay (best training episode)</div>'
                    '<div class="policy-grid">'
                    f'{instruction_panel}'
                    f'<img src="data:image/gif;base64,{state.policy_gif_b64}" '
                    'style="max-width:100%;border-radius:6px;border:1px solid #1e3a5f;'
                    'margin-top:6px;display:block" alt="policy replay gif">'
                    '</div>'
                )
            else:
                policy_section = (
                    '<div class="policy-lbl">Optimal policy replay (best training episode)</div>'
                    f'<img src="data:image/gif;base64,{state.policy_gif_b64}" '
                    'style="max-width:100%;border-radius:6px;border:1px solid #1e3a5f;'
                    'margin-top:6px;display:block" alt="policy replay gif">'
                )
        elif state.policy_frames:
            # ANSI / text frames — show as a cycling JS slideshow
            frames_json = json.dumps(state.policy_frames)
            card_id = f"pf_{state.agent_id}"
            if instruction_panel:
                policy_section = (
                    '<div class="policy-lbl">Optimal policy replay (best training episode)</div>'
                    '<div class="policy-grid">'
                    f'{instruction_panel}'
                    f'<div class="policy-pre" id="{card_id}"></div>'
                    '</div>'
                    f'<script>(function(){{'
                    f'var frames={frames_json},i=0,el=document.getElementById("{card_id}");'
                    f'if(!el)return;'
                    f'el.textContent=frames[0];'
                    f'setInterval(function(){{i=(i+1)%frames.length;el.textContent=frames[i];}},600);'
                    f'}})();</script>'
                )
            else:
                policy_section = (
                    '<div class="policy-lbl">Optimal policy replay (best training episode)</div>'
                    f'<div class="policy-pre" id="{card_id}"></div>'
                    f'<script>(function(){{'
                    f'var frames={frames_json},i=0,el=document.getElementById("{card_id}");'
                    f'if(!el)return;'
                    f'el.textContent=frames[0];'
                    f'setInterval(function(){{i=(i+1)%frames.length;el.textContent=frames[i];}},600);'
                    f'}})();</script>'
                )
        elif state.policy_text:
            policy_section = (
                '<div class="policy-lbl">Current policy snapshot</div>'
                f'<div class="policy-pre">{html.escape(state.policy_text)}</div>'
            )
    elif state.policy_text:
        policy_section = (
            '<div class="policy-lbl">Current policy snapshot</div>'
            f'<div class="policy-pre">{html.escape(state.policy_text)}</div>'
        )

    return f"""
<div class="card">
  <div class="card-header">
    <span style="font-size:1rem;font-weight:700;color:#f1f5f9">{html.escape(state.agent_id)}</span>
    <span class="env-badge">{html.escape(state.env_id)}</span>
    <span class="type-badge">{html.escape(state.agent_type)}</span>
        {lang_tag}
        {instr_tag}
    {done_badge}
  </div>
  <div style="margin-bottom:10px">{_progress_bar(state.completed, state.n_episodes)}</div>
  <div class="metrics">
    <div class="metric">
      <span class="metric-val">{state.last_reward:+.3f}</span>
      <span class="metric-lbl">Last reward</span>
    </div>
    <div class="metric">
      <span class="metric-val">{mean_recent:+.3f}</span>
      <span class="metric-lbl">Mean (last 50)</span>
    </div>
    <div class="metric">
      <span class="metric-val">{state.best_reward:+.3f}</span>
      <span class="metric-lbl">Best</span>
    </div>
    <div class="metric">
      <span class="metric-val">{state.epsilon:.4f}</span>
      <span class="metric-lbl">ε (epsilon)</span>
    </div>
    <div class="metric">
      <span class="metric-val">{elapsed_str}</span>
      <span class="metric-lbl">Elapsed</span>
    </div>
  </div>
  <div class="chart-wrap">{_sparkline_svg(state.episode_rewards)}</div>
  {policy_section}
</div>
"""


def _build_html(version: int, states: list[_AgentState]) -> str:
    if not states:
        body = '<div class="idle">No training runs yet. Start one with <code>rl_train_agent()</code>.</div>'
    else:
        body = "\n".join(_render_agent_card(s) for s in reversed(states))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLIP Training Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<h1>RLIP Training Dashboard <span id="refresh-dot"></span></h1>
<div class="subtitle">Auto-refreshes every 1.5 s &nbsp;·&nbsp; <span id="ts">v{version}</span></div>
<div id="main">{body}</div>
<script>{_JS}</script>
</body>
</html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — serves the dashboard page and a version endpoint."""

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # silence request logging

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/version":
            payload = json.dumps({"version": dashboard.get_version()}).encode()
            self._respond(200, "application/json", payload)

        elif path in ("/", "/index.html"):
            ver, states = dashboard.snapshot()
            page = _build_html(ver, states).encode("utf-8")
            self._respond(200, "text/html; charset=utf-8", page)

        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


# ── Server lifecycle ──────────────────────────────────────────────────────────

_server_lock = threading.Lock()
_server_thread: Optional[threading.Thread] = None
_server_port: Optional[int] = None
_PREFERRED_PORT = 7432  # default; falls back to OS-assigned if taken


def start_dashboard(port: int = _PREFERRED_PORT) -> str:
    """
    Start the dashboard HTTP server in a daemon thread (idempotent).

    Returns the URL, e.g. ``http://localhost:7432``.
    """
    global _server_thread, _server_port

    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return f"http://localhost:{_server_port}"

        # Try requested port; fall back to any free port
        for try_port in (port, 0):
            try:
                server = HTTPServer(("localhost", try_port), _Handler)
                break
            except OSError:
                if try_port == 0:
                    raise

        actual_port = server.server_address[1]
        _server_port = actual_port

        def _serve() -> None:
            server.serve_forever()

        _server_thread = threading.Thread(target=_serve, daemon=True, name="rlip-dashboard")
        _server_thread.start()
        return f"http://localhost:{actual_port}"


def is_running() -> bool:
    with _server_lock:
        return _server_thread is not None and _server_thread.is_alive()


def dashboard_url() -> Optional[str]:
    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return f"http://localhost:{_server_port}"
        return None
