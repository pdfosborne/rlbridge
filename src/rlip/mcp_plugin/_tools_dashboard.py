"""
Dashboard MCP tools for the RLIP plugin.

Tools:
  rl_start_dashboard  - start the live training dashboard web server
  rl_stop_dashboard   - (no-op - server is daemon-threaded; provided for symmetry)
"""

from __future__ import annotations

from ._dashboard import dashboard as _dash, start_dashboard, dashboard_url, is_running
from ._state import mcp


@mcp.tool()
def rl_start_dashboard(port: int = 7432) -> str:
    """
    Start the live RL training dashboard web server.

    The dashboard is served at http://localhost:<port> and automatically
    updates every 1.5 seconds while agents are being trained.  The server
    runs in a background daemon thread and stops automatically when the
    Python process exits.

    Calling this tool again while the server is already running is safe - it
    returns the existing URL without restarting.

    Features visible in the browser
    --------------------------------
    * Per-agent reward curve (episode history sparkline with rolling mean)
    * Progress bar (completed / total episodes)
    * Key metrics: last reward, mean of last 50 episodes, best reward, ε
    * Elapsed training time
    * Optional policy text snapshot (when available)

    Parameters
    ----------
    port:
        TCP port to listen on (default 7432).  If that port is already in
        use by something else, the server falls back to any free port.

    Returns
    -------
    The URL where the dashboard is accessible, e.g.
    ``http://localhost:7432``.  Open this in any browser.
    """
    url = start_dashboard(port=port)
    already = "already running" if is_running() else "started"  # misleading if just started, but harmless
    return (
        f"Dashboard {already} at {url}\n\n"
        "Open that URL in your browser.  The page refreshes every 1.5 s "
        "and shows live reward curves as soon as you call rl_train_agent()."
    )


@mcp.tool()
def rl_dashboard_status() -> str:
    """
    Report whether the live training dashboard is currently running.

    Returns the URL if running, or a message indicating it is not started.
    Call rl_start_dashboard() to start it.
    """
    url = dashboard_url()
    if url:
        _, states = _dash.snapshot()
        n_agents = len(states)
        running  = sum(1 for s in states if s.completed < s.n_episodes)
        done     = n_agents - running
        return (
            f"Dashboard is running at {url}\n"
            f"  Agents tracked: {n_agents}  ({running} training, {done} complete)"
        )
    return (
        "Dashboard is not running.  "
        "Call rl_start_dashboard() to start it."
    )
