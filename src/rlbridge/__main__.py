"""
RL Bridge CLI  (``rlbridge`` command)
================================
Provides these commands:

    rlbridge server              - start the HTTP JSON-RPC server
    rlbridge mcp                 - start the MCP stdio plugin
    rlbridge install-claude          - configure RL Bridge in ~/.claude.json  (Claude Code)
    rlbridge install-claude-desktop  - configure RL Bridge in Claude Desktop GUI
    rlbridge install-codex           - configure RL Bridge in ~/.codex/config.toml  (Codex CLI)
    rlbridge install-opencode        - configure RL Bridge in ~/.config/opencode/config.json  (OpenCode)
    rlbridge install-lmstudio        - configure RL Bridge in LM Studio (~/.lmstudio/mcp.json)
    rlbridge install-cursor          - configure RL Bridge in ~/.cursor/mcp.json  (Cursor)
    rlbridge install-windsurf        - configure RL Bridge in Windsurf (~/.codeium/windsurf/mcp_config.json)
    rlbridge agent               - run an Ollama / OpenAI-compatible agent loop
    rlbridge catalog             - list the public environment catalog
    rlbridge install             - download a public environment from GitHub
    rlbridge uninstall           - remove a cached public environment
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rlbridge",
    help="Reinforcement Learning Bridge CLI",
    pretty_exceptions_enable=False,
)
console = Console()


# ── rlbridge server ───────────────────────────────────────────────────────────────

@app.command("server")
def server_app(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8765, help="Bind port"),
    auto_register: bool = typer.Option(
        True,
        "--auto-register/--no-auto-register",
        help="Auto-register all installed Gymnasium environments",
    ),
    log_level: str = typer.Option("info", help="Uvicorn log level"),
    max_instances: int = typer.Option(64, help="Maximum concurrent environment instances"),
) -> None:
    """Start the RL Bridge HTTP server (JSON-RPC 2.0 over HTTP)."""
    import uvicorn

    from .environments.registry import registry
    from .server.rlbridge_server import create_app

    if auto_register:
        with console.status("Auto-registering Gymnasium environments…"):
            count = registry.auto_register_gymnasium()
        console.print(f"[green]Registered {count} Gymnasium environments.[/green]")

    app_instance = create_app(env_registry=registry, max_instances=max_instances)

    console.print(
        f"[bold green]RL Bridge Server[/bold green] listening on "
        f"[bold]http://{host}:{port}[/bold]"
    )
    console.print(f"  Docs:       http://{host}:{port}/docs")
    console.print(f"  Health:     http://{host}:{port}/health")
    console.print(f"  RPC:   POST http://{host}:{port}/rpc")

    uvicorn.run(app_instance, host=host, port=port, log_level=log_level)


# ── rlbridge mcp ──────────────────────────────────────────────────────────────────

@app.command("mcp")
def mcp_command() -> None:
    """Start the RL Bridge MCP stdio plugin (used by Claude Code)."""
    from .mcp_plugin.plugin import main
    main()


# ── rlbridge install-claude ───────────────────────────────────────────────────────

@app.command("install-claude")
def install_claude(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to claude config file (defaults to ~/.claude.json)",
    ),
) -> None:
    """
    Add RL Bridge to your Claude Code MCP configuration (~/.claude.json).

    After running this command, restart Claude Code and RL Bridge tools will be
    available in your Claude sessions.
    """
    target = config_path or Path.home() / ".claude.json"

    # Use the home directory as cwd so relative paths (e.g. .rlbridge/) resolve
    # to a user-writable location regardless of how the client launches the server.
    cwd = str(Path.home())

    if use_script:
        entry: dict = {"command": "rlbridge-mcp", "type": "stdio", "cwd": cwd}
    else:
        entry = {
            "command": command,
            "args": ["-m", "rlbridge.mcp_plugin"],
            "type": "stdio",
            "cwd": cwd,
        }

    # Load or create config
    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: {target} contains invalid JSON - creating fresh config.[/yellow]")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["rlbridge"] = entry

    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RL Bridge MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlbridge": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Claude Code\n"
        "  2. Open a new conversation\n"
        "  3. Ask Claude to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-codex ───────────────────────────────────────────────────────

@app.command("install-codex")
def install_codex(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to Codex config.toml (defaults to ~/.codex/config.toml)",
    ),
) -> None:
    """
    Add RL Bridge to your Codex CLI MCP configuration (~/.codex/config.toml).

    After running this command, restart Codex and RL Bridge tools will be
    available in your Codex sessions.
    """
    target = config_path or Path.home() / ".codex" / "config.toml"

    if use_script:
        toml_block = "[mcp_servers.rlbridge]\ncommand = \"rlbridge-mcp\"\n"
    else:
        args_toml = json.dumps(["-m", "rlbridge.mcp_plugin"])
        toml_block = (
            "[mcp_servers.rlbridge]\n"
            f'command = "{command}"\n'
            f"args = {args_toml}\n"
        )

    existing = target.read_text() if target.exists() else ""

    section_pattern = re.compile(
        r"\[mcp_servers\.rlbridge\][^\[]*", re.DOTALL
    )
    if section_pattern.search(existing):
        updated = section_pattern.sub(toml_block, existing)
    else:
        separator = "\n" if existing and not existing.endswith("\n") else ""
        updated = existing + separator + "\n" + toml_block

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated)
    console.print(f"[green]✓ RL Bridge MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(toml_block)
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Codex\n"
        "  2. Open a new conversation\n"
        "  3. Ask Codex to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-opencode ─────────────────────────────────────────────────────

@app.command("install-opencode")
def install_opencode(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to OpenCode config (defaults to ~/.config/opencode/config.json)",
    ),
) -> None:
    """
    Add RL Bridge to your OpenCode MCP configuration (~/.config/opencode/config.json).

    After running this command, restart OpenCode and RL Bridge tools will be
    available in your OpenCode sessions.
    """
    target = config_path or Path.home() / ".config" / "opencode" / "config.json"

    if use_script:
        cmd_array = ["rlbridge-mcp"]
    else:
        cmd_array = [command, "-m", "rlbridge.mcp_plugin"]

    entry: dict = {"type": "local", "command": cmd_array}

    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(
                f"[yellow]Warning: {target} contains invalid JSON - creating fresh config.[/yellow]"
            )

    config.setdefault("mcp", {})
    config["mcp"]["rlbridge"] = entry

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RL Bridge MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcp": {"rlbridge": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart OpenCode\n"
        "  2. Open a new conversation\n"
        "  3. Ask OpenCode to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-claude-desktop ───────────────────────────────────────────────

def _find_claude_desktop_config_windows() -> list[Path]:
    """
    Locate all claude_desktop_config.json paths on Windows.

    Claude Desktop is distributed two ways on Windows:
      1. Classic installer  → %APPDATA%\\Claude\\claude_desktop_config.json
      2. Microsoft Store    → %LOCALAPPDATA%\\Packages\\Claude_<id>\\LocalCache\\Roaming\\Claude\\claude_desktop_config.json

    Returns every path that either already exists or whose parent package
    folder exists, so callers can write to all of them.
    """
    import glob

    found: list[Path] = []

    # Classic installer path
    appdata = os.environ.get("APPDATA", "")
    classic = Path(appdata) / "Claude" / "claude_desktop_config.json"
    if classic.exists() or classic.parent.exists():
        found.append(classic)

    # Microsoft Store sandbox path
    localappdata = os.environ.get("LOCALAPPDATA", "")
    # Glob for any matching package folder (publisher hash varies)
    pkg_pattern = str(Path(localappdata) / "Packages" / "Claude_*" / "LocalCache" / "Roaming" / "Claude")
    for pkg_dir in glob.glob(pkg_pattern):
        store_path = Path(pkg_dir) / "claude_desktop_config.json"
        if store_path not in found:
            found.append(store_path)

    # If nothing found at all, fall back to classic (will be created)
    if not found:
        found.append(classic)

    return found


@app.command("install-claude-desktop")
def install_claude_desktop(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        None,  # None means "auto"
        "--use-script/--use-module",
        help=(
            "Use the 'rlbridge-mcp' script entry-point instead of 'python -m'. "
            "Defaults to True on Windows (avoids PATH issues in Claude Desktop) "
            "and False elsewhere."
        ),
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to the Claude Desktop config file (auto-detected per OS by default)",
    ),
) -> None:
    """
    Add RL Bridge to your Claude Desktop MCP configuration (GUI app, not Claude Code).

    The config path is chosen automatically:
      macOS   - ~/Library/Application Support/Claude/claude_desktop_config.json
      Windows - %APPDATA%\\Claude\\  OR  Microsoft Store sandbox path (auto-detected)
      Linux   - ~/.config/Claude/claude_desktop_config.json

    After running this command, restart Claude Desktop and RL Bridge tools will be
    available in your conversations.
    """
    import platform

    system = platform.system()

    # Default use_script to True on Windows: Claude Desktop (both classic and
    # Store installs) does not inherit the user's PATH, so bare 'python' fails.
    # Using the full path to the rlbridge-mcp.exe script is more reliable.
    if use_script is None:
        use_script = system == "Windows"

    # On Windows, resolve the rlbridge-mcp script to its full absolute path so
    # Claude Desktop can find it without needing PATH.
    # Always set cwd to the user home dir so .rlbridge/ writes land somewhere
    # writable - Claude Desktop otherwise defaults to C:\Windows\System32.
    cwd = str(Path.home())

    if use_script:
        if system == "Windows":
            import shutil
            script_path = shutil.which("rlbridge-mcp") or "rlbridge-mcp"
            entry: dict = {"command": script_path, "cwd": cwd}
        else:
            entry = {"command": "rlbridge-mcp", "cwd": cwd}
    else:
        entry = {"command": command, "args": ["-m", "rlbridge.mcp_plugin"], "cwd": cwd}

    # Build list of config paths to write
    if config_path is not None:
        config_paths: list[Path] = [config_path]
    elif system == "Darwin":
        config_paths = [Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"]
    elif system == "Windows":
        config_paths = _find_claude_desktop_config_windows()
    else:
        config_paths = [Path.home() / ".config" / "Claude" / "claude_desktop_config.json"]

    written: list[Path] = []
    for cp in config_paths:
        config: dict = {}
        if cp.exists():
            try:
                config = json.loads(cp.read_text())
            except json.JSONDecodeError:
                console.print(f"[yellow]Warning: {cp} contains invalid JSON - creating fresh config.[/yellow]")
        config.setdefault("mcpServers", {})
        config["mcpServers"]["rlbridge"] = entry
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(config, indent=2))
        written.append(cp)
        console.print(f"[green]✓ RL Bridge MCP server added to {cp}[/green]")

    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlbridge": entry}}, indent=2))
    if len(written) > 1:
        console.print(f"\n[dim](Written to {len(written)} config files - classic install and Microsoft Store install)[/dim]")
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Fully quit Claude Desktop (right-click system tray icon → Quit)\n"
        "  2. Reopen Claude Desktop\n"
        "  3. Go to Settings → Developer to confirm 'rlbridge' is listed\n"
        "  4. Ask Claude to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-lmstudio ─────────────────────────────────────────────────────

@app.command("install-lmstudio")
def install_lmstudio(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to LM Studio's mcp.json (auto-detected per OS by default)",
    ),
) -> None:
    """
    Add RL Bridge to your LM Studio MCP configuration (mcp.json).

    LM Studio follows Cursor's mcp.json notation.  The config path is chosen
    automatically:
      macOS   - ~/Library/Application Support/LM Studio/mcp.json
      Windows - %APPDATA%\\LM Studio\\mcp.json
      Linux   - ~/.lmstudio/mcp.json

    After running this command, restart LM Studio and RL Bridge tools will be
    available in your chat sessions.
    """
    import platform

    if config_path is None:
        system = platform.system()
        if system == "Darwin":
            config_path = Path.home() / "Library" / "Application Support" / "LM Studio" / "mcp.json"
        elif system == "Windows":
            appdata = os.environ.get("APPDATA", "")
            config_path = Path(appdata) / "LM Studio" / "mcp.json"
        else:
            config_path = Path.home() / ".lmstudio" / "mcp.json"

    if use_script:
        entry: dict = {"command": "rlbridge-mcp", "type": "stdio"}
    else:
        entry = {"command": command, "args": ["-m", "rlbridge.mcp_plugin"], "type": "stdio"}

    config: dict = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: {config_path} contains invalid JSON - creating fresh config.[/yellow]")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["rlbridge"] = entry

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RL Bridge MCP server added to {config_path}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlbridge": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart LM Studio\n"
        "  2. Open the chat panel (Program tab → enable MCP)\n"
        "  3. Ask your model to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-cursor ───────────────────────────────────────────────────────

@app.command("install-cursor")
def install_cursor(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to Cursor's mcp.json (defaults to ~/.cursor/mcp.json)",
    ),
) -> None:
    """
    Add RL Bridge to your Cursor MCP configuration (~/.cursor/mcp.json).

    After running this command, restart Cursor and RL Bridge tools will be
    available in Agent mode.
    """
    target = config_path or Path.home() / ".cursor" / "mcp.json"

    if use_script:
        entry: dict = {"command": "rlbridge-mcp"}
    else:
        entry = {"command": command, "args": ["-m", "rlbridge.mcp_plugin"]}

    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: {target} contains invalid JSON - creating fresh config.[/yellow]")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["rlbridge"] = entry

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RL Bridge MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlbridge": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Cursor\n"
        "  2. Open a new Agent chat\n"
        "  3. Ask the agent to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge install-windsurf ─────────────────────────────────────────────────────

@app.command("install-windsurf")
def install_windsurf(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlbridge-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to Windsurf's MCP config (defaults to ~/.codeium/windsurf/mcp_config.json)",
    ),
) -> None:
    """
    Add RL Bridge to your Windsurf MCP configuration (~/.codeium/windsurf/mcp_config.json).

    After running this command, restart Windsurf and RL Bridge tools will be
    available in Cascade agent sessions.
    """
    target = config_path or Path.home() / ".codeium" / "windsurf" / "mcp_config.json"

    if use_script:
        entry: dict = {"command": "rlbridge-mcp"}
    else:
        entry = {"command": command, "args": ["-m", "rlbridge.mcp_plugin"]}

    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: {target} contains invalid JSON - creating fresh config.[/yellow]")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["rlbridge"] = entry

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RL Bridge MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlbridge": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Windsurf\n"
        "  2. Open a new Cascade session\n"
        "  3. Ask the agent to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlbridge list ─────────────────────────────────────────────────────────────────

@app.command("list")
def list_environments(
    all_envs: bool = typer.Option(
        False,
        "--all",
        help="Show all installed Gymnasium environments, not just the common presets",
    ),
) -> None:
    """List available RL environments."""
    from .environments.registry import registry

    if all_envs:
        with console.status("Discovering all Gymnasium environments…"):
            registry.auto_register_gymnasium()

    envs = registry.list_environments()

    table = Table(title=f"RL Bridge Environments ({len(envs)} total)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Render modes")
    table.add_column("Max steps")
    table.add_column("Reward threshold")

    for env in sorted(envs, key=lambda e: e.env_id):
        table.add_row(
            env.env_id,
            ", ".join(env.render_modes) or "–",
            str(env.max_episode_steps) if env.max_episode_steps else "–",
            str(env.reward_threshold) if env.reward_threshold is not None else "–",
        )

    console.print(table)


# ── rlbridge agent ───────────────────────────────────────────────────────────────

@app.command("agent")
def agent_command(
    task: str = typer.Argument(
        ...,
        help="Natural-language task, e.g. 'Run CartPole for 50 steps with a random policy'",
    ),
    model: str = typer.Option(
        "llama3.1",
        "--model", "-m",
        help="Model name served at the endpoint (e.g. llama3.1, mistral, gpt-4o)",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434/v1",
        "--base-url", "-u",
        help="OpenAI-compatible chat completions base URL. Ollama default: http://localhost:11434/v1",
    ),
    api_key: str = typer.Option(
        "ollama",
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="API key (required for OpenAI, any string works for Ollama)",
    ),
    max_iter: int = typer.Option(
        64,
        "--max-iter",
        help="Maximum tool-calling rounds before giving up",
    ),
    auto_register: bool = typer.Option(
        True,
        "--auto-register/--no-auto-register",
        help="Auto-register all installed Gymnasium environments before starting",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show each tool call and result as they happen",
    ),
) -> None:
    """
    Run an Ollama (or any OpenAI-compatible) model as an RL Bridge agent.

    The model receives all RL Bridge tools as OpenAI function definitions and can
    interact with RL environments through function-calling.

    Examples
    --------
        # Ollama (default)
        rlbridge agent "Run CartPole-v1 with a random policy"
        rlbridge agent --model mistral "List available environments"

        # OpenAI
        rlbridge agent --base-url https://api.openai.com/v1 --model gpt-4o --api-key $OPENAI_API_KEY "Run LunarLander"
    """
    from .adapters.openai_agent import rlbridgeAgent
    from .environments.registry import registry

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if auto_register:
        with console.status("Auto-registering Gymnasium environments…"):
            try:
                count = registry.auto_register_gymnasium()
                console.print(f"[dim]Registered {count} additional Gymnasium environments.[/dim]")
            except Exception as exc:
                console.print(f"[yellow]auto_register_gymnasium: {exc}[/yellow]")

    console.print(f"[bold]RL Bridge Agent[/bold]  model=[cyan]{model}[/cyan]  endpoint=[dim]{base_url}[/dim]")
    console.print(f"[bold]Task:[/bold] {task}\n")

    with console.status("Thinking…", spinner="dots"):
        try:
            with rlbridgeAgent(
                base_url=base_url,
                model=model,
                api_key=api_key,
                max_iterations=max_iter,
                registry=registry,
            ) as agent:
                answer = agent.run(task)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            raise typer.Exit(1)

    console.print("[bold green]Answer:[/bold green]")
    console.print(answer)


import logging


# ── rlbridge catalog ────────────────────────────────────────────────────────────

@app.command("catalog")
def catalog_command(
    tags: str = typer.Option("", help="Comma-separated tags to filter by"),
    installed_only: bool = typer.Option(False, "--installed", help="Only show installed envs"),
) -> None:
    """List public environments available in the RL Bridge catalog."""
    from .public_registry import public_registry

    tag_filter = [t.strip() for t in tags.split(",") if t.strip()]
    entries = public_registry.list_catalog(tags=tag_filter or None, installed_only=installed_only)

    if not entries:
        console.print("No entries match the given filters.")
        return

    table = Table(title=f"RL Bridge Public Catalog ({len(entries)} environments)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Installed", justify="center")
    table.add_column("Tags")
    table.add_column("Description")

    for e in entries:
        installed = "✓" if public_registry.is_installed(e["env_id"]) else ""
        table.add_row(
            e["env_id"],
            f"[green]{installed}[/green]" if installed else "",
            ", ".join(e.get("tags", [])),
            e.get("description", "")[:80],
        )

    console.print(table)
    console.print(
        "\nInstall with: [bold]rlbridge install <env_id>[/bold]\n"
        "Use in code:  [bold]from rlbridge.public_registry import public_registry[/bold]"
    )


# ── rlbridge install ───────────────────────────────────────────────────────────

@app.command("install")
def install_command(
    env_id: str = typer.Argument(..., help="Public environment ID from the catalog, e.g. 'Sailing-v0'"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if already installed"),
) -> None:
    """
    Download a public RL environment from GitHub and cache it locally.

    After installing, the environment is available to any RL Bridge agent or server
    via its env_id (no extra configuration needed - it auto-loads from cache).

    Examples
    --------
        rlbridge install Sailing-v0
        rlbridge install Sailing-Hard-v0
    """
    from .public_registry import public_registry

    if public_registry.is_installed(env_id) and not force:
        console.print(f"[green]✓[/green] '{env_id}' is already installed.")
        console.print("Use [bold]--force[/bold] to re-download.")
        return

    try:
        entry = public_registry.get_entry(env_id)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    deps = entry.get("extra_dependencies", [])
    if deps:
        console.print(f"[yellow]Note:[/yellow] This environment requires extra packages: {deps}")
        console.print(f"Install them with: [bold]pip install {' '.join(deps)}[/bold]\n")

    with console.status(f"Downloading '{env_id}' from GitHub…"):
        try:
            path = public_registry.install(env_id, force=force)
        except FileNotFoundError as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Install error:[/red] {exc}")
            raise typer.Exit(1)

    console.print(f"[green]✓ Installed[/green] '{env_id}' → {path}")
    console.print(
        f"\nUse it:\n"
        f"  Python: [bold]public_registry.install_and_load('{env_id}', registry)[/bold]\n"
        f"  Agent:  [bold]rlbridge agent \"Run an episode of {env_id}\"[/bold]"
    )


# ── rlbridge uninstall ───────────────────────────────────────────────────────

@app.command("uninstall")
def uninstall_command(
    env_id: str = typer.Argument(..., help="Public environment ID to remove"),
) -> None:
    """Remove a cached public environment from local storage."""
    from .public_registry import public_registry

    if not public_registry.is_installed(env_id):
        console.print(f"'{env_id}' is not installed.")
        return

    removed = public_registry.uninstall(env_id)
    if removed:
        console.print(f"[green]✓ Removed[/green] '{env_id}'")
    else:
        console.print(f"[yellow]Nothing removed for '{env_id}'.[/yellow]")


if __name__ == "__main__":
    app()
