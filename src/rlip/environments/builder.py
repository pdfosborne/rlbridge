"""
Environment Builder
====================
Fluent API for defining, caching, and registering new RLIP environments.

The builder handles three concerns in one place:

* **Metadata** — description, tags, namespace, episode limits.
* **Local cache** — persists everything to ``~/.rlip/environments/<env_id>/`` so the
  environment can be reloaded in future sessions without repeating the build
  steps.
* **Language translation** — attach a named, custom, or LLM-generated
  :class:`~rlip.language_translation.base.LanguageTranslator`.  Generated
  translators are saved as a standalone Python module inside the cache.

Quick start
-----------
**Wrap a Gymnasium environment:**

::

    from rlip.environments.builder import EnvironmentBuilder

    built = (
        EnvironmentBuilder("FrozenLake-Custom-v0")
        .from_gymnasium("FrozenLake-v1")
        .with_metadata(
            description="FrozenLake with a custom translator.",
            tags=["grid", "discrete"],
        )
        .with_translator("FrozenLake-v1")   # use any registered translator
        .build()
    )

    env = built.create()
    built.register()          # loads into the running RLIP registry

**Use an LLM-generated translator:**

::

    import openai

    def my_llm(prompt: str) -> str:
        return openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

    built = (
        EnvironmentBuilder("FrozenLake-Custom-v0")
        .from_gymnasium("FrozenLake-v1")
        .with_metadata(description="FrozenLake with LLM translation.",
                       tags=["grid", "discrete"])
        .with_llm_translator(
            llm_fn=my_llm,
            env_context="4×4 frozen lake grid.  State is an integer 0–15.",
            n_samples=20,
        )
        .build()
    )

**Reload from cache:**

::

    from rlip.environments.builder import BuiltEnvironment

    built = BuiltEnvironment.load("FrozenLake-Custom-v0")
    built.register()

**Bulk-load all cached environments at startup:**

::

    from rlip.environments.builder import load_cached_environments

    load_cached_environments()   # auto-registers into the global registry

"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..language_translation.base import LanguageTranslator
from ..protocol.messages import EnvironmentInfo
from .base import RLIPEnvironment, RLIPEnvironmentFactory

# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR: Path = Path.home() / ".rlip" / "environments"
_USER_CATALOG: Path = Path.home() / ".rlip" / "catalog.json"


# ── Spec dataclass ────────────────────────────────────────────────────────────

@dataclass
class EnvSpec:
    """
    Serialisable description of a custom environment.

    Persisted as ``spec.json`` inside the environment's cache directory
    and mirrored into the user catalog (``~/.rlip/catalog.json``).
    """
    env_id: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    namespace: str = "custom"
    max_episode_steps: Optional[int] = None
    render_modes: list[str] = field(default_factory=list)
    #: Describes how to recreate the factory.
    source: dict[str, Any] = field(default_factory=dict)
    #: Language translation metadata (may be empty).
    language_translation: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvSpec":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


# ── Factory recreation helpers ────────────────────────────────────────────────

def _make_factory(spec: EnvSpec) -> RLIPEnvironmentFactory:
    """Recreate the env factory from persisted *spec.source*."""
    src = spec.source
    src_type = src.get("type", "")

    if src_type == "gymnasium":
        from .gymnasium_adapter import GymnasiumFactory
        base = GymnasiumFactory(src["gym_env_id"])
        # Only wrap with override if the env_id or metadata differs
        if spec.env_id != src["gym_env_id"] or spec.description or spec.namespace != "gymnasium":
            return _OverrideMetadataFactory(
                inner=base,
                env_id=spec.env_id,
                description=spec.description,
                tags=spec.tags,
                namespace=spec.namespace,
                max_episode_steps=spec.max_episode_steps,
                render_modes=spec.render_modes,
            )
        return base

    if src_type == "class":
        module_path = src.get("module_path")
        class_name = src.get("class_name")
        init_kwargs = src.get("init_kwargs", {})
        if module_path and class_name:
            module = _load_module_from_path("_custom_factory", Path(module_path))
            factory_cls = getattr(module, class_name)
            return factory_cls(**init_kwargs)

    raise ValueError(
        f"Cannot recreate factory for env_id={spec.env_id!r}: "
        f"unsupported source type {src_type!r}.  "
        "Provide the factory directly via BuiltEnvironment(factory=...) "
        "or re-run the builder."
    )


def _load_module_from_path(name: str, path: Path):
    """Dynamically import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_translator_from_spec(
    lt_spec: dict[str, Any],
    cache_dir: Path,
) -> Optional[LanguageTranslator]:
    """Recreate a :class:`LanguageTranslator` from its persisted spec."""
    lt_type = lt_spec.get("type", "")

    if not lt_type:
        return None

    if lt_type == "named":
        from ..language_translation import get_translator
        return get_translator(lt_spec["name"])

    if lt_type == "generated":
        module_path = Path(lt_spec.get("module_path", ""))
        if not module_path.is_absolute():
            module_path = cache_dir / module_path
        class_name = lt_spec.get("class_name", "")
        if not module_path.exists():
            return None
        module = _load_module_from_path(f"_gen_translator_{class_name}", module_path)
        cls = getattr(module, class_name, None)
        return cls() if cls is not None else None

    if lt_type == "class":
        module_name = lt_spec.get("module_name")  # dotted import path
        module_path = lt_spec.get("module_path")
        class_name = lt_spec.get("class_name")
        if not class_name:
            return None
        if module_name:
            try:
                import importlib as _il
                module = _il.import_module(module_name)
                cls = getattr(module, class_name, None)
                return cls() if cls is not None else None
            except Exception:
                pass
        if module_path:
            try:
                module = _load_module_from_path(f"_translator_{class_name}", Path(module_path))
                cls = getattr(module, class_name, None)
                return cls() if cls is not None else None
            except Exception:
                pass
        return None

    if lt_type == "inline_instance":
        # Cannot deserialise arbitrary objects — return None
        return None

    return None


# ── GymnasiumFactory override helpers ─────────────────────────────────────────

def _gymnasium_factory_with_overrides(
    gym_env_id: str,
    override_id: Optional[str],
    tags: list[str],
    description: str,
    namespace: str,
    max_episode_steps: Optional[int],
    render_modes: list[str],
) -> RLIPEnvironmentFactory:
    """Build a GymnasiumFactory wrapped with overridden EnvironmentInfo metadata."""
    from .gymnasium_adapter import GymnasiumFactory

    base = GymnasiumFactory(gym_env_id, tags=tags, description=description)
    if not override_id and namespace == "gymnasium" and not max_episode_steps and not render_modes:
        return base
    return _OverrideMetadataFactory(
        inner=base,
        env_id=override_id or gym_env_id,
        description=description or base.env_info.description,
        tags=tags or base.env_info.tags,
        namespace=namespace,
        max_episode_steps=max_episode_steps or base.env_info.max_episode_steps,
        render_modes=render_modes or base.env_info.render_modes,
    )


# ── BuiltEnvironment ───────────────────────────────────────────────────────────

class BuiltEnvironment:
    """
    A finalised custom environment with optional language translation.

    Returned by :meth:`EnvironmentBuilder.build` and
    :meth:`BuiltEnvironment.load`.

    Attributes
    ----------
    spec : EnvSpec
        Full persisted metadata for this environment.
    factory : RLIPEnvironmentFactory
        Factory used to create environment instances.
    translator : LanguageTranslator or None
        Language translator, if one was configured.
    cache_path : Path
        Directory where the environment's files are stored on disk.
    """

    def __init__(
        self,
        spec: EnvSpec,
        factory: RLIPEnvironmentFactory,
        translator: Optional[LanguageTranslator] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.spec = spec
        self.factory = factory
        self.translator = translator
        self.cache_path = cache_path or (_DEFAULT_CACHE_DIR / spec.env_id)

    # ── Environment creation ──────────────────────────────────────────────────

    def create(self, render_mode: Optional[str] = None, **kwargs: Any) -> RLIPEnvironment:
        """Instantiate a new environment instance from this factory."""
        return self.factory.create(render_mode=render_mode, **kwargs)

    # ── Registry integration ──────────────────────────────────────────────────

    def register(
        self,
        reg=None,
        *,
        register_translator: bool = True,
    ) -> None:
        """
        Register the environment (and optionally its translator) at runtime.

        Parameters
        ----------
        reg:
            :class:`~rlip.environments.registry.EnvironmentRegistry` to
            register into.  Defaults to the global ``registry`` singleton.
        register_translator:
            If *True* and a translator is attached, add it to the global
            ``TRANSLATORS`` dict so protocols can look it up automatically.
        """
        if reg is None:
            from .registry import registry as _registry
            reg = _registry
        reg.register(self.factory)

        if register_translator and self.translator is not None:
            from ..language_translation import TRANSLATORS
            TRANSLATORS[self.spec.env_id] = self.translator

    # ── Catalog ───────────────────────────────────────────────────────────────

    def update_catalog(self, catalog_path: Optional[Path] = None) -> Path:
        """
        Write (or update) this environment's entry in the user catalog.

        Parameters
        ----------
        catalog_path:
            Path to the target ``catalog.json`` file.  Defaults to
            ``~/.rlip/catalog.json``.

        Returns
        -------
        Path
            The catalog file that was written.
        """
        path = catalog_path or _USER_CATALOG
        path.parent.mkdir(parents=True, exist_ok=True)

        catalog: dict[str, Any] = {"catalog_version": "0.2", "environments": []}
        if path.exists():
            try:
                catalog = json.loads(path.read_text())
            except Exception:
                pass

        envs: list[dict[str, Any]] = catalog.get("environments", [])
        # Replace existing entry if present, otherwise append
        updated = False
        entry = self.spec.to_dict()
        for i, e in enumerate(envs):
            if e.get("env_id") == self.spec.env_id:
                envs[i] = entry
                updated = True
                break
        if not updated:
            envs.append(entry)
        catalog["environments"] = envs

        path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return path

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, cache_dir: Optional[Path] = None) -> Path:
        """
        Persist spec + translator to the local cache.

        The cache directory layout::

            ~/.rlip/environments/<env_id>/
                spec.json         — environment metadata
                translator.py     — generated translator module (if present)

        Returns
        -------
        Path
            The environment's cache directory.
        """
        env_dir = (cache_dir or _DEFAULT_CACHE_DIR) / self.spec.env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = env_dir

        # Write spec
        (env_dir / "spec.json").write_text(
            json.dumps(self.spec.to_dict(), indent=2), encoding="utf-8"
        )
        return env_dir

    @classmethod
    def load(
        cls,
        env_id: str,
        cache_dir: Optional[Path] = None,
    ) -> "BuiltEnvironment":
        """
        Reload a previously built environment from the local cache.

        Parameters
        ----------
        env_id:
            The environment identifier used when it was built.
        cache_dir:
            Root cache directory.  Defaults to ``~/.rlip/environments/``.

        Returns
        -------
        BuiltEnvironment

        Raises
        ------
        FileNotFoundError
            If no cached entry exists for *env_id*.
        """
        env_dir = (cache_dir or _DEFAULT_CACHE_DIR) / env_id
        spec_path = env_dir / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"No cached environment found for '{env_id}' "
                f"(looked in {env_dir})"
            )

        spec = EnvSpec.from_dict(json.loads(spec_path.read_text()))
        factory = _make_factory(spec)
        translator = _load_translator_from_spec(
            spec.language_translation, env_dir
        )
        return cls(spec=spec, factory=factory, translator=translator,
                   cache_path=env_dir)

    def __repr__(self) -> str:
        has_translator = self.translator is not None
        return (
            f"BuiltEnvironment(env_id={self.spec.env_id!r}, "
            f"namespace={self.spec.namespace!r}, "
            f"translator={has_translator}, "
            f"cache={self.cache_path})"
        )


# ── EnvironmentBuilder ────────────────────────────────────────────────────────

class EnvironmentBuilder:
    """
    Fluent builder for creating, caching, and registering new environments.

    Start by specifying the environment source (:meth:`from_gymnasium` or
    :meth:`from_factory`), then layer on metadata and a language translator,
    and finally call :meth:`build` to produce a :class:`BuiltEnvironment`.

    Parameters
    ----------
    env_id:
        The identifier this environment will be registered under.  If
        wrapping a Gymnasium environment with a *different* ID, the Gymnasium
        env is still created internally with the original Gymnasium ID.
    """

    def __init__(self, env_id: str) -> None:
        self._env_id = env_id
        self._description = ""
        self._tags: list[str] = []
        self._namespace = "custom"
        self._max_episode_steps: Optional[int] = None
        self._render_modes: list[str] = []

        # Source
        self._gym_env_id: Optional[str] = None
        self._gym_kwargs: dict[str, Any] = {}
        self._custom_factory: Optional[RLIPEnvironmentFactory] = None
        self._source_meta: dict[str, Any] = {}

        # Translator
        self._translator: Optional[LanguageTranslator] = None
        self._translator_spec: dict[str, Any] = {}
        self._llm_fn: Optional[Callable[[str], str]] = None
        self._llm_kwargs: dict[str, Any] = {}

    # ── Source ────────────────────────────────────────────────────────────────

    def from_gymnasium(
        self,
        gym_env_id: str,
        **gym_kwargs: Any,
    ) -> "EnvironmentBuilder":
        """
        Wrap an existing Gymnasium environment.

        Parameters
        ----------
        gym_env_id:
            The Gymnasium environment identifier (e.g. ``"FrozenLake-v1"``).
        **gym_kwargs:
            Extra keyword arguments forwarded to ``gym.make``.
        """
        self._gym_env_id = gym_env_id
        self._gym_kwargs = gym_kwargs
        self._source_meta = {
            "type": "gymnasium",
            "gym_env_id": gym_env_id,
            "gym_kwargs": gym_kwargs,
        }
        return self

    def from_factory(
        self,
        factory: RLIPEnvironmentFactory,
        *,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None,
        init_kwargs: Optional[dict[str, Any]] = None,
    ) -> "EnvironmentBuilder":
        """
        Use an existing :class:`~rlip.environments.base.RLIPEnvironmentFactory`.

        For the factory to be reloadable from cache, provide *module_path*
        and *class_name* so it can be reimported on the next session.
        Otherwise the cached entry will reference a ``"class"`` source that
        requires the same Python path to be available.

        Parameters
        ----------
        factory:
            A ready-to-use factory instance.
        module_path:
            Absolute path to the ``.py`` file that defines the factory class.
        class_name:
            Class name of the factory (e.g. ``"MyGridFactory"``).
        init_kwargs:
            Keyword arguments to pass to the class constructor on reload.
        """
        self._custom_factory = factory
        self._source_meta = {
            "type": "class",
            "module_path": module_path or "",
            "class_name": class_name or type(factory).__name__,
            "init_kwargs": init_kwargs or {},
        }
        return self

    # ── Metadata ──────────────────────────────────────────────────────────────

    def with_metadata(
        self,
        description: str = "",
        tags: Optional[list[str]] = None,
        namespace: str = "custom",
        max_episode_steps: Optional[int] = None,
        render_modes: Optional[list[str]] = None,
    ) -> "EnvironmentBuilder":
        """
        Set environment metadata that appears in registry listings.

        Parameters
        ----------
        description:
            Human-readable description of the environment.
        tags:
            Searchable tags (e.g. ``["grid", "discrete"]``).
        namespace:
            Namespace shown in registry listings.  Defaults to ``"custom"``.
        max_episode_steps:
            Hard step limit per episode.
        render_modes:
            Supported render modes (e.g. ``["rgb_array"]``).
        """
        self._description = description
        self._tags = tags or []
        self._namespace = namespace
        self._max_episode_steps = max_episode_steps
        self._render_modes = render_modes or []
        return self

    # ── Translator ────────────────────────────────────────────────────────────

    def with_translator(
        self,
        translator: "str | LanguageTranslator",
    ) -> "EnvironmentBuilder":
        """
        Attach a language translator to the environment.

        Parameters
        ----------
        translator:
            Either a :class:`~rlip.language_translation.base.LanguageTranslator`
            instance, or a string env_id / translator name to look up from the
            global ``TRANSLATORS`` registry (e.g. ``"Sailing-v0"``).
        """
        if isinstance(translator, str):
            from ..language_translation import get_translator
            looked_up = get_translator(translator)
            if looked_up is None:
                raise ValueError(
                    f"No translator registered for {translator!r}.  "
                    "Pass a LanguageTranslator instance directly, or use "
                    "with_llm_translator() to generate one."
                )
            self._translator = looked_up
            self._translator_spec = {"type": "named", "name": translator}
        else:
            self._translator = translator
            # Try to record the class location for persistence
            cls = type(translator)
            mod = getattr(cls, "__module__", None) or ""
            if mod and not mod.startswith("<"):
                spec_file = getattr(
                    sys.modules.get(mod), "__file__", None
                )
                self._translator_spec = {
                    "type": "class",
                    "module_name": mod,
                    "module_path": spec_file or "",
                    "class_name": cls.__name__,
                }
            else:
                # Inline instance — not reloadable from cache
                self._translator_spec = {"type": "inline_instance"}
        return self

    def with_llm_translator(
        self,
        llm_fn: Callable[[str], str],
        env_context: str = "",
        n_samples: int = 30,
        seed: Optional[int] = None,
        refine_threshold: int = 20,
    ) -> "EnvironmentBuilder":
        """
        Configure an LLM-generated language translator.

        The translator is built at :meth:`build` time by sampling states from
        the environment and running the two-stage LLM pipeline.  The resulting
        rule function is saved as ``translator.py`` in the cache directory.

        Parameters
        ----------
        llm_fn:
            LLM callable ``(prompt: str) -> str``.
        env_context:
            Short English description of the environment, given to the LLM
            to guide description and rule-synthesis prompts.
        n_samples:
            Number of unique states to sample.
        seed:
            Random seed for reproducible state sampling.
        refine_threshold:
            Number of LLM-answered cache entries before automatic refinement.
        """
        self._llm_fn = llm_fn
        self._llm_kwargs = {
            "env_context": env_context,
            "n_samples": n_samples,
            "seed": seed,
            "refine_threshold": refine_threshold,
        }
        # Spec will be filled in at build() once we have the env instance
        return self

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        cache_dir: Optional[Path] = None,
        *,
        auto_register: bool = True,
        update_catalog: bool = True,
    ) -> BuiltEnvironment:
        """
        Finalise the environment: create the factory, run LLM translation
        generation (if configured), persist to the local cache, and
        optionally register into the global registry and user catalog.

        Parameters
        ----------
        cache_dir:
            Override the default cache root (``~/.rlip/environments/``).
        auto_register:
            If *True*, immediately call
            :meth:`BuiltEnvironment.register` after building so the
            environment is available in the current session.
        update_catalog:
            If *True*, write the entry to ``~/.rlip/catalog.json``.

        Returns
        -------
        BuiltEnvironment
        """
        # ── 1. Build factory ──────────────────────────────────────────────────
        factory = self._build_factory()

        # ── 2. Build translator (LLM path) ────────────────────────────────────
        env_dir = (cache_dir or _DEFAULT_CACHE_DIR) / self._env_id
        translator, translator_spec = self._build_translator(factory, env_dir)

        # ── 3. Assemble spec ──────────────────────────────────────────────────
        spec = EnvSpec(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace=self._namespace,
            max_episode_steps=self._max_episode_steps,
            render_modes=self._render_modes,
            source=self._source_meta,
            language_translation=translator_spec,
        )

        result = BuiltEnvironment(
            spec=spec,
            factory=factory,
            translator=translator,
            cache_path=env_dir,
        )

        # ── 4. Cache to disk ──────────────────────────────────────────────────
        result.save(cache_dir)

        # ── 5. Catalog ────────────────────────────────────────────────────────
        if update_catalog:
            result.update_catalog()

        # ── 6. Register ───────────────────────────────────────────────────────
        if auto_register:
            result.register()

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_factory(self) -> RLIPEnvironmentFactory:
        if self._custom_factory is not None:
            # Wrap with overridden metadata if the user supplied any
            if any([self._description, self._tags, self._max_episode_steps,
                    self._render_modes, self._namespace != "custom"]):
                return _OverrideMetadataFactory(
                    inner=self._custom_factory,
                    env_id=self._env_id,
                    description=self._description,
                    tags=self._tags,
                    namespace=self._namespace,
                    max_episode_steps=self._max_episode_steps,
                    render_modes=self._render_modes,
                )
            return self._custom_factory

        if self._gym_env_id is not None:
            return _gymnasium_factory_with_overrides(
                gym_env_id=self._gym_env_id,
                override_id=self._env_id,
                tags=self._tags,
                description=self._description,
                namespace=self._namespace,
                max_episode_steps=self._max_episode_steps,
                render_modes=self._render_modes,
            )

        raise RuntimeError(
            "No environment source specified.  Call from_gymnasium() or "
            "from_factory() before build()."
        )

    def _build_translator(
        self,
        factory: RLIPEnvironmentFactory,
        env_dir: Path,
    ) -> tuple[Optional[LanguageTranslator], dict[str, Any]]:
        """Build and (if LLM) save the translator.  Returns (translator, spec)."""
        # Non-LLM path — translator already set
        if self._llm_fn is None:
            return self._translator, self._translator_spec

        # LLM path
        from ..language_translation.generator import TranslatorGenerator, _to_class_name

        env = factory.create()
        try:
            gen = TranslatorGenerator(
                env=env,
                llm_fn=self._llm_fn,
                **self._llm_kwargs,
                # env_id for naming
            )
            # Expose env_id to generator
            gen._env_id = self._env_id
            translator = gen.build()
        finally:
            try:
                env.close()
            except Exception:
                pass

        # Save the rule function as a standalone Python module
        env_dir.mkdir(parents=True, exist_ok=True)
        class_name = _to_class_name(self._env_id)
        module_path = env_dir / "translator.py"
        translator.save_code(module_path, class_name=class_name)

        translator_spec = {
            "type": "generated",
            "module_path": "translator.py",   # relative to env_dir
            "class_name": class_name,
            "env_context": self._llm_kwargs.get("env_context", ""),
        }
        return translator, translator_spec


# ── Metadata override factory wrapper ─────────────────────────────────────────

class _OverrideMetadataFactory(RLIPEnvironmentFactory):
    """Wraps an existing factory, replacing its EnvironmentInfo metadata."""

    def __init__(
        self,
        inner: RLIPEnvironmentFactory,
        env_id: str,
        description: str,
        tags: list[str],
        namespace: str,
        max_episode_steps: Optional[int],
        render_modes: list[str],
    ) -> None:
        self._inner = inner
        self._env_info = EnvironmentInfo(
            env_id=env_id,
            description=description,
            tags=tags,
            namespace=namespace,
            max_episode_steps=max_episode_steps,
            render_modes=render_modes,
        )

    @property
    def env_info(self) -> EnvironmentInfo:
        return self._env_info

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> RLIPEnvironment:
        env = self._inner.create(render_mode=render_mode, **kwargs)
        env._env_id = self._env_info.env_id
        return env


# ── Bulk loader ───────────────────────────────────────────────────────────────

def load_cached_environments(
    reg=None,
    cache_dir: Optional[Path] = None,
    *,
    register_translators: bool = True,
) -> list[BuiltEnvironment]:
    """
    Load and register all environments previously built with
    :class:`EnvironmentBuilder`.

    Parameters
    ----------
    reg:
        :class:`~rlip.environments.registry.EnvironmentRegistry` to register
        into.  Defaults to the global ``registry`` singleton.
    cache_dir:
        Root cache directory.  Defaults to ``~/.rlip/environments/``.
    register_translators:
        Whether to also register translators into the global ``TRANSLATORS``
        dict.

    Returns
    -------
    list[BuiltEnvironment]
        Successfully loaded environments (failures are logged and skipped).
    """
    import logging
    log = logging.getLogger(__name__)

    root = cache_dir or _DEFAULT_CACHE_DIR
    if not root.exists():
        return []

    built_envs: list[BuiltEnvironment] = []
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        spec_path = env_dir / "spec.json"
        if not spec_path.exists():
            continue
        try:
            env_id = env_dir.name
            built = BuiltEnvironment.load(env_id, cache_dir=root)
            built.register(reg=reg, register_translator=register_translators)
            built_envs.append(built)
        except Exception as exc:
            log.warning("Failed to load cached environment %r: %s", env_dir.name, exc)

    return built_envs


__all__ = [
    "EnvSpec",
    "BuiltEnvironment",
    "EnvironmentBuilder",
    "load_cached_environments",
]
