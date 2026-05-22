"""
Public Environment Installer
==============================
Downloads environment files from GitHub and stores them in a local cache
at ``~/.rlip/public_envs/<env_id>/``.

Each cached environment directory contains:
  engine.py          - the downloaded environment source file(s)
  meta.json          - copy of the catalog entry at install time

No file is executed at install time — the engine module is only imported
when the environment is first instantiated.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Cache directory ───────────────────────────────────────────────────────────

CACHE_ROOT = Path.home() / ".rlip" / "public_envs"


def cache_dir(env_id: str) -> Path:
    """Return (and create) the local cache directory for *env_id*."""
    d = CACHE_ROOT / env_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_installed(env_id: str) -> bool:
    """True if the environment has been downloaded to the local cache."""
    meta = CACHE_ROOT / env_id / "meta.json"
    return meta.exists()


def list_installed() -> list[str]:
    """Return env IDs of all locally cached public environments."""
    if not CACHE_ROOT.exists():
        return []
    return [
        d.name
        for d in CACHE_ROOT.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    ]


# ── Download ──────────────────────────────────────────────────────────────────

def _raw_url(repo: str, branch: str, path: str) -> str:
    """Build a GitHub raw content URL.  Only github.com repos are supported."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def install(entry: dict[str, Any], force: bool = False) -> Path:
    """
    Download a public environment described by *entry* (a catalog dict) into
    the local cache.  Returns the cache directory path.

    Parameters
    ----------
    entry:
        A single environment entry from ``catalog.json``.
    force:
        Re-download even if already installed.
    """
    env_id: str = entry["env_id"]
    target = cache_dir(env_id)
    meta_path = target / "meta.json"

    if meta_path.exists() and not force:
        log.debug("'%s' already installed at %s — skipping.", env_id, target)
        return target

    source: dict[str, Any] = entry["source"]
    source_type = source.get("type")
    if source_type == "github":
        return _install_github(entry, target, meta_path, source, force)
    if source_type == "plugin":
        return _install_plugin(entry, target, meta_path, source, force)

    raise ValueError(
        f"Unsupported source type '{source_type}' for '{env_id}'. "
        "Supported: 'github', 'plugin'."
    )


def _install_github(
    entry: dict[str, Any],
    target: Path,
    meta_path: Path,
    source: dict[str, Any],
    force: bool,
) -> Path:
    env_id: str = entry["env_id"]
    repo: str = source["repo"]
    branch: str = source.get("branch", "main")
    files: list[dict[str, str]] = source["files"]

    log.info("Installing '%s' from github.com/%s …", env_id, repo)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for file_spec in files:
            remote_path: str = file_spec["path"]
            local_name: str = file_spec.get("save_as", Path(remote_path).name)
            url = _raw_url(repo, branch, remote_path)

            log.debug("  GET %s", url)
            resp = client.get(url)

            if resp.status_code == 404:
                raise FileNotFoundError(
                    f"Remote file not found: {url}\n"
                    f"Check that '{repo}' has '{remote_path}' on branch '{branch}'."
                )
            resp.raise_for_status()

            dest = target / local_name
            dest.write_bytes(resp.content)
            log.info("  Saved %s → %s", remote_path, dest)

    meta_path.write_text(json.dumps(entry, indent=2))
    log.info("'%s' installed to %s", env_id, target)
    return target


def _install_plugin(
    entry: dict[str, Any],
    target: Path,
    meta_path: Path,
    source: dict[str, Any],
    force: bool,
) -> Path:
    import subprocess
    import sys

    env_id: str = entry["env_id"]
    install_url = source.get("install_url") or source.get("package")
    if not install_url:
        raise ValueError(
            f"Plugin source for '{env_id}' must include 'install_url' or 'package'."
        )

    log.info("Installing plugin package for '%s' via pip …", env_id)
    cmd = [sys.executable, "-m", "pip", "install"]
    if force:
        cmd.append("--force-reinstall")
    cmd.append(install_url)
    subprocess.run(cmd, check=True)

    meta_path.write_text(json.dumps(entry, indent=2))
    log.info("'%s' plugin installed for %s", source.get("package", install_url), env_id)
    return target


def uninstall(env_id: str) -> bool:
    """Remove a cached environment.  Returns True if something was removed."""
    import shutil
    target = CACHE_ROOT / env_id
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


# ── Dynamic import ────────────────────────────────────────────────────────────

def load_engine_class(env_id: str, source: dict[str, Any]) -> type:
    """
    Dynamically import the engine module from the local cache and return
    the Engine class referenced by ``source["engine_class"]``.
    """
    target = cache_dir(env_id)
    module_name = source.get("engine_module", "engine")
    class_name  = source.get("engine_class", "Engine")

    module_file = target / f"{module_name}.py"
    if not module_file.exists():
        raise FileNotFoundError(
            f"Engine file not found: {module_file}\n"
            f"Run 'rlip install {env_id}' first."
        )

    # Build a unique module name to avoid collisions across envs
    qualified = f"rlip_public.{env_id.replace('-', '_')}.{module_name}"

    if qualified in sys.modules:
        mod = sys.modules[qualified]
    else:
        spec = importlib.util.spec_from_file_location(qualified, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec from {module_file}")
        # Ensure the env's own directory is on sys.path so relative imports work
        pkg_dir = str(target)
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{class_name}' not found in {module_file}. "
            f"Available names: {[n for n in dir(mod) if not n.startswith('_')]}"
        )
    return cls
