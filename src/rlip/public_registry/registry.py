"""
RLIP Public Registry
======================
Manages the catalog of publicly available environments and provides a
unified API for installing, loading, and registering them.

Catalog location
----------------
Built-in catalog: ``rlip/public_registry/catalog.json``  (ships with RLIP)
User catalog:     ``~/.rlip/catalog.json``                (add your own entries)

The user catalog is merged over the built-in catalog, so entries with the
same ``env_id`` in the user catalog take precedence.

Usage
-----
    from rlip.public_registry import public_registry

    # See what's available
    entries = public_registry.list_catalog()

    # Download an environment
    public_registry.install("Sailing-v0")

    # Load all installed envs into the RLIP registry
    public_registry.load_installed(registry)

    # Or install + load in one call
    public_registry.install_and_load("Sailing-v0", registry)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_BUILTIN_CATALOG = Path(__file__).parent / "catalog.json"
_USER_CATALOG    = Path.home() / ".rlip" / "catalog.json"


class PublicRegistry:
    """Manages the catalog and installation of public RLIP environments."""

    def __init__(self) -> None:
        self._catalog: dict[str, dict[str, Any]] = {}
        self._reload_catalog()

    # ── Catalog management ────────────────────────────────────────────────────

    def _reload_catalog(self) -> None:
        """Merge built-in and user catalogs into self._catalog."""
        self._catalog = {}

        for path in (_BUILTIN_CATALOG, _USER_CATALOG):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                for entry in data.get("environments", []):
                    env_id = entry.get("env_id")
                    if env_id:
                        self._catalog[env_id] = entry
            except Exception as exc:
                log.warning("Failed to load catalog from %s: %s", path, exc)

    def list_catalog(
        self,
        tags: list[str] | None = None,
        installed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return catalog entries, optionally filtered.

        Parameters
        ----------
        tags:
            If given, only return entries containing ALL of these tags.
        installed_only:
            If True, only return entries that are already downloaded.
        """
        from .installer import is_installed
        entries = list(self._catalog.values())

        if tags:
            entries = [e for e in entries if set(tags).issubset(set(e.get("tags", [])))]
        if installed_only:
            entries = [e for e in entries if is_installed(e["env_id"])]

        return entries

    def get_entry(self, env_id: str) -> dict[str, Any]:
        """Return the catalog entry for *env_id*, raising KeyError if absent."""
        if env_id not in self._catalog:
            raise KeyError(
                f"'{env_id}' is not in the RLIP public catalog.\n"
                f"Available: {list(self._catalog.keys())}"
            )
        return self._catalog[env_id]

    def add_to_user_catalog(self, entry: dict[str, Any]) -> None:
        """
        Append *entry* to the user catalog at ``~/.rlip/catalog.json``.
        Creates the file if it doesn't exist.
        """
        _USER_CATALOG.parent.mkdir(parents=True, exist_ok=True)

        if _USER_CATALOG.exists():
            data = json.loads(_USER_CATALOG.read_text())
        else:
            data = {"catalog_version": "0.1", "environments": []}

        # Remove existing entry with same env_id
        data["environments"] = [
            e for e in data["environments"]
            if e.get("env_id") != entry["env_id"]
        ]
        data["environments"].append(entry)
        _USER_CATALOG.write_text(json.dumps(data, indent=2))
        self._reload_catalog()

    # ── Install ───────────────────────────────────────────────────────────────

    def install(self, env_id: str, force: bool = False) -> Path:
        """
        Download the environment files for *env_id* from GitHub and cache them.

        Parameters
        ----------
        env_id:
            Must be a known catalog entry.
        force:
            Re-download even if already installed.

        Returns the local cache directory path.
        """
        from .installer import install, is_installed
        entry = self.get_entry(env_id)

        if is_installed(env_id) and not force:
            log.info("'%s' is already installed.", env_id)
            from .installer import cache_dir
            return cache_dir(env_id)

        return install(entry, force=force)

    def uninstall(self, env_id: str) -> bool:
        """Remove cached files for *env_id*. Returns True if removed."""
        from .installer import uninstall
        return uninstall(env_id)

    # ── Load into RLIP registry ───────────────────────────────────────────────

    def load_installed(
        self,
        registry: "EnvironmentRegistry",  # type: ignore[name-defined]
    ) -> list[str]:
        """
        Register all locally installed public environments into *registry*.
        Returns the list of env_ids that were successfully loaded.
        """
        from .installer import list_installed
        loaded: list[str] = []

        for env_id in list_installed():
            try:
                self._load_one(env_id, registry)
                loaded.append(env_id)
            except Exception as exc:
                log.warning("Failed to load public env '%s': %s", env_id, exc)

        return loaded

    def install_and_load(
        self,
        env_id: str,
        registry: "EnvironmentRegistry",  # type: ignore[name-defined]
        force: bool = False,
    ) -> None:
        """Download *env_id* if needed and register it in *registry*."""
        self.install(env_id, force=force)
        self._load_one(env_id, registry)

    def _load_one(
        self,
        env_id: str,
        registry: "EnvironmentRegistry",  # type: ignore[name-defined]
    ) -> None:
        """Register one installed env into *registry*."""
        if env_id not in self._catalog:
            # Env was installed via a previous catalog; try loading meta from disk
            from .installer import CACHE_ROOT
            meta_path = CACHE_ROOT / env_id / "meta.json"
            if meta_path.exists():
                entry = json.loads(meta_path.read_text())
                self._catalog[env_id] = entry
            else:
                raise KeyError(f"No catalog entry for '{env_id}'")

        entry = self._catalog[env_id]
        adapter_type = entry.get("source", {}).get("adapter", "sailing")

        if adapter_type == "sailing":
            from ..environments.predefined.sailing import SailingFactory
            factory = SailingFactory(
                env_id=entry["env_id"],
                y_limit=entry.get("default_setup_info", {}).get("y_limit", 25.0),
                x_limit=entry.get("default_setup_info", {}).get("x_limit", 10.0),
                obs_precision=entry.get("default_setup_info", {}).get("obs_precision", 4),
                supervised_rewards=entry.get("default_setup_info", {}).get("supervised_rewards", False),
                tags=entry.get("tags", []),
                max_episode_steps=entry.get("max_episode_steps", 200),
                description=entry.get("description", ""),
            )
        elif adapter_type == "flesh_and_blood":
            from ..environments.plugins import load_plugin_environments

            load_plugin_environments(registry)
            if env_id not in registry:
                package = entry.get("source", {}).get("package", "flesh-and-blood-rlip")
                raise ValueError(
                    f"Plugin '{package}' is installed but '{env_id}' is not registered."
                )
            log.info("Registered public env '%s' via plugin.", env_id)
            return
        else:
            raise ValueError(
                f"Unknown adapter type '{adapter_type}' for '{env_id}'. "
                "Supported: 'sailing', 'flesh_and_blood'."
            )

        registry.register(factory)
        log.info("Registered public env '%s'.", env_id)
        self._register_translator(env_id, entry)

    def _register_translator(self, env_id: str, entry: dict[str, Any]) -> None:
        """Auto-register the default translator for *env_id* into TRANSLATORS."""
        lt = entry.get("language_translation")
        if not lt or not lt.get("default"):
            return
        try:
            import importlib
            from ..language_translation import TRANSLATORS
            module = importlib.import_module(lt["module"])
            cls = getattr(module, lt["class"])
            TRANSLATORS[env_id] = cls()
            log.debug("Registered translator '%s' for '%s'.", lt["class"], env_id)
        except Exception as exc:
            log.warning("Could not register translator for '%s': %s", env_id, exc)

    # ── Convenience ───────────────────────────────────────────────────────────

    def is_installed(self, env_id: str) -> bool:
        from .installer import is_installed
        return is_installed(env_id)

    def get_translator(self, env_id: str, name: str | None = None):
        """
        Return the language translator for *env_id* from the catalog.

        Parameters
        ----------
        env_id:
            Environment ID to look up.
        name:
            Translator name to select from ``language_translation.available``.
            Defaults to the ``default`` entry in the catalog.

        Returns
        -------
        LanguageTranslator | None
            The instantiated translator, or ``None`` if none is defined.
        """
        entry = self._catalog.get(env_id)
        if not entry:
            return None
        lt = entry.get("language_translation")
        if not lt:
            return None

        selected = name or lt.get("default")
        if selected not in lt.get("available", []):
            raise KeyError(
                f"Translator '{selected}' not listed in available translators "
                f"for '{env_id}': {lt.get('available')}"
            )

        import importlib
        module = importlib.import_module(lt["module"])
        cls = getattr(module, lt["class"])
        return cls()

    def list_translators(self, env_id: str) -> list[str]:
        """Return the names of available translators for *env_id*."""
        entry = self._catalog.get(env_id, {})
        return entry.get("language_translation", {}).get("available", [])


# Global singleton
public_registry = PublicRegistry()
