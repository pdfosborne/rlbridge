"""
RLIP CLI  (``rlip`` command)
==============================
Provides these commands:

    rlip server              – start the HTTP JSON-RPC server
    rlip mcp                 – start the MCP stdio plugin
    rlip install-claude      – configure RLIP in ~/.claude.json  (Claude Code)
    rlip install-codex       – configure RLIP in ~/.codex/config.toml  (Codex CLI)
    rlip install-opencode    – configure RLIP in ~/.config/opencode/config.json  (OpenCode)
    rlip agent               – run an Ollama / OpenAI-compatible agent loop
    rlip catalog             – list the public environment catalog
    rlip install             – download a public environment from GitHub
    rlip uninstall           – remove a cached public environment
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
    name="rlip",
    help="Reinforcement Learning Interaction Protocol CLI",
    pretty_exceptions_enable=False,
)
console = Console()


# ── rlip server ───────────────────────────────────────────────────────────────

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
    """Start the RLIP HTTP server (JSON-RPC 2.0 over HTTP)."""
    import uvicorn

    from .environments.registry import registry
    from .server.rlip_server import create_app

    if auto_register:
        with console.status("Auto-registering Gymnasium environments…"):
            count = registry.auto_register_gymnasium()
        console.print(f"[green]Registered {count} Gymnasium environments.[/green]")

    app_instance = create_app(env_registry=registry, max_instances=max_instances)

    console.print(
        f"[bold green]RLIP Server[/bold green] listening on "
        f"[bold]http://{host}:{port}[/bold]"
    )
    console.print(f"  Docs:       http://{host}:{port}/docs")
    console.print(f"  Health:     http://{host}:{port}/health")
    console.print(f"  RPC:   POST http://{host}:{port}/rpc")

    uvicorn.run(app_instance, host=host, port=port, log_level=log_level)


# ── rlip mcp ──────────────────────────────────────────────────────────────────

@app.command("mcp")
def mcp_command() -> None:
    """Start the RLIP MCP stdio plugin (used by Claude Code)."""
    from .mcp_plugin.plugin import main
    main()


# ── rlip install-claude ───────────────────────────────────────────────────────

@app.command("install-claude")
def install_claude(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlip-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to claude config file (defaults to ~/.claude.json)",
    ),
) -> None:
    """
    Add RLIP to your Claude Code MCP configuration (~/.claude.json).

    After running this command, restart Claude Code and RLIP tools will be
    available in your Claude sessions.
    """
    target = config_path or Path.home() / ".claude.json"

    if use_script:
        entry: dict = {"command": "rlip-mcp", "type": "stdio"}
    else:
        entry = {
            "command": command,
            "args": ["-m", "rlip.mcp_plugin"],
            "type": "stdio",
        }

    # Load or create config
    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: {target} contains invalid JSON – creating fresh config.[/yellow]")

    config.setdefault("mcpServers", {})
    config["mcpServers"]["rlip"] = entry

    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RLIP MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcpServers": {"rlip": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Claude Code\n"
        "  2. Open a new conversation\n"
        "  3. Ask Claude to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlip install-codex ───────────────────────────────────────────────────────

@app.command("install-codex")
def install_codex(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlip-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to Codex config.toml (defaults to ~/.codex/config.toml)",
    ),
) -> None:
    """
    Add RLIP to your Codex CLI MCP configuration (~/.codex/config.toml).

    After running this command, restart Codex and RLIP tools will be
    available in your Codex sessions.
    """
    target = config_path or Path.home() / ".codex" / "config.toml"

    if use_script:
        toml_block = "[mcp_servers.rlip]\ncommand = \"rlip-mcp\"\n"
    else:
        args_toml = json.dumps(["-m", "rlip.mcp_plugin"])
        toml_block = (
            "[mcp_servers.rlip]\n"
            f'command = "{command}"\n'
            f"args = {args_toml}\n"
        )

    existing = target.read_text() if target.exists() else ""

    section_pattern = re.compile(
        r"\[mcp_servers\.rlip\][^\[]*", re.DOTALL
    )
    if section_pattern.search(existing):
        updated = section_pattern.sub(toml_block, existing)
    else:
        separator = "\n" if existing and not existing.endswith("\n") else ""
        updated = existing + separator + "\n" + toml_block

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated)
    console.print(f"[green]✓ RLIP MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(toml_block)
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart Codex\n"
        "  2. Open a new conversation\n"
        "  3. Ask Codex to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlip install-opencode ─────────────────────────────────────────────────────

@app.command("install-opencode")
def install_opencode(
    command: str = typer.Option(
        sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    ),
    use_script: bool = typer.Option(
        False,
        "--use-script/--use-module",
        help="Use the 'rlip-mcp' script instead of 'python -m'",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Path to OpenCode config (defaults to ~/.config/opencode/config.json)",
    ),
) -> None:
    """
    Add RLIP to your OpenCode MCP configuration (~/.config/opencode/config.json).

    After running this command, restart OpenCode and RLIP tools will be
    available in your OpenCode sessions.
    """
    target = config_path or Path.home() / ".config" / "opencode" / "config.json"

    if use_script:
        cmd_array = ["rlip-mcp"]
    else:
        cmd_array = [command, "-m", "rlip.mcp_plugin"]

    entry: dict = {"type": "local", "command": cmd_array}

    config: dict = {}
    if target.exists():
        try:
            config = json.loads(target.read_text())
        except json.JSONDecodeError:
            console.print(
                f"[yellow]Warning: {target} contains invalid JSON – creating fresh config.[/yellow]"
            )

    config.setdefault("mcp", {})
    config["mcp"]["rlip"] = entry

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2))
    console.print(f"[green]✓ RLIP MCP server added to {target}[/green]")
    console.print("\nConfiguration written:")
    console.print(json.dumps({"mcp": {"rlip": entry}}, indent=2))
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Restart OpenCode\n"
        "  2. Open a new conversation\n"
        "  3. Ask OpenCode to 'run a CartPole episode' to verify the plugin works"
    )


# ── rlip list ─────────────────────────────────────────────────────────────────

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

    table = Table(title=f"RLIP Environments ({len(envs)} total)")
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


# ── rlip agent ───────────────────────────────────────────────────────────────

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
    Run an Ollama (or any OpenAI-compatible) model as an RLIP agent.

    The model receives all RLIP tools as OpenAI function definitions and can
    interact with RL environments through function-calling.

    Examples
    --------
        # Ollama (default)
        rlip agent "Run CartPole-v1 with a random policy"
        rlip agent --model mistral "List available environments"

        # OpenAI
        rlip agent --base-url https://api.openai.com/v1 --model gpt-4o --api-key $OPENAI_API_KEY "Run LunarLander"
    """
    from .adapters.openai_agent import RLIPAgent
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

    console.print(f"[bold]RLIP Agent[/bold]  model=[cyan]{model}[/cyan]  endpoint=[dim]{base_url}[/dim]")
    console.print(f"[bold]Task:[/bold] {task}\n")

    with console.status("Thinking…", spinner="dots"):
        try:
            with RLIPAgent(
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


# ── rlip catalog ────────────────────────────────────────────────────────────

@app.command("catalog")
def catalog_command(
    tags: str = typer.Option("", help="Comma-separated tags to filter by"),
    installed_only: bool = typer.Option(False, "--installed", help="Only show installed envs"),
) -> None:
    """List public environments available in the RLIP catalog."""
    from .public_registry import public_registry

    tag_filter = [t.strip() for t in tags.split(",") if t.strip()]
    entries = public_registry.list_catalog(tags=tag_filter or None, installed_only=installed_only)

    if not entries:
        console.print("No entries match the given filters.")
        return

    table = Table(title=f"RLIP Public Catalog ({len(entries)} environments)")
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
        "\nInstall with: [bold]rlip install <env_id>[/bold]\n"
        "Use in code:  [bold]from rlip.public_registry import public_registry[/bold]"
    )


# ── rlip install ───────────────────────────────────────────────────────────

@app.command("install")
def install_command(
    env_id: str = typer.Argument(..., help="Public environment ID from the catalog, e.g. 'Sailing-v0'"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if already installed"),
) -> None:
    """
    Download a public RL environment from GitHub and cache it locally.

    After installing, the environment is available to any RLIP agent or server
    via its env_id (no extra configuration needed — it auto-loads from cache).

    Examples
    --------
        rlip install Sailing-v0
        rlip install Sailing-Hard-v0
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
        f"  Agent:  [bold]rlip agent \"Run an episode of {env_id}\"[/bold]"
    )


# ── rlip uninstall ───────────────────────────────────────────────────────

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
