"""
Custom environment creation MCP tools for the RLIP plugin.

Tools: rl_create_environment_from_code, rl_get_environment_template,
       rl_validate_environment_code, rl_delete_custom_environment
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Optional

from ._state import (
    _catalog_path,
    _custom_env_cache_root,
    _custom_translators,
    _env_agents_dir,
    _env_cache_dir,
    _env_renders_dir,
    mcp,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_TEMPLATE = '''\
"""
{env_id} - Custom RLIP Environment
====================================
Describe your environment here.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from rlbridge.environments.base import RLIPEnvironment, RLIPEnvironmentFactory
from rlbridge.protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
)


class {class_name}Env(RLIPEnvironment):
    """
    Custom environment: {env_id}.

    Observations : describe what the agent sees (e.g. integer, list, dict).
    Actions      : describe the action space (e.g. 0=left 1=right).
    Rewards      : describe the reward structure.
    Termination  : describe when an episode ends.
    """

    def __init__(self) -> None:
        # TODO: initialise your environment state here
        self._state: Any = None
        self._step_count: int = 0
        self._initialized: bool = False

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        rng = random.Random(seed)
        # TODO: initialise your state using rng
        self._state = 0
        self._step_count = 0
        self._initialized = True
        return ResetResult(observation=self._state, info={{}})

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        self._step_count += 1

        # TODO: implement transition dynamics
        # action: int (0 or 1 for this skeleton)
        terminated = False
        truncated = False
        reward = 0.0

        return StepResult(
            observation=self._state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={{}},
        )

    def close(self) -> None:
        self._initialized = False

    # ── Spaces ────────────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> DiscreteSpace:
        # TODO: update to match your observation structure
        return DiscreteSpace(n=10, start=0)

    @property
    def action_space(self) -> DiscreteSpace:
        # TODO: update to match your action structure
        return DiscreteSpace(n=2)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> RenderResult:
        # TODO: return a text description of the current state
        return RenderResult(mode="ansi", text=f"state={{self._state}}")

    def sample_action(self) -> Any:
        return random.randrange(2)  # TODO: match your action_space.n


class {class_name}Factory(RLIPEnvironmentFactory):
    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id="{env_id}",
            description="{description}",
            tags={tags_repr},
            namespace="custom",
            max_episode_steps={max_episode_steps},
            render_modes=["ansi"],
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> {class_name}Env:
        return {class_name}Env()


# Module-level singleton used by rl_load_cached_environments / registry
FACTORY = {class_name}Factory()
'''

_VALIDATION_SCRIPT = '''\
import sys, json, traceback

def _run(module_path, env_id):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_env_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    factory = getattr(module, "FACTORY", None)
    if factory is None:
        # Try to find any RLIPEnvironmentFactory subclass
        from rlbridge.environments.base import RLIPEnvironmentFactory
        for name in dir(module):
            obj = getattr(module, name)
            try:
                if isinstance(obj, RLIPEnvironmentFactory):
                    factory = obj
                    break
            except Exception:
                pass

    if factory is None:
        return {"ok": False, "error": "No module-level FACTORY instance found."}

    errors = []
    info_ok = False
    try:
        info = factory.env_info
        assert info.env_id, "env_info.env_id is empty"
        info_ok = True
    except Exception as exc:
        errors.append(f"env_info failed: {exc}")

    try:
        env = factory.create()
    except Exception as exc:
        errors.append(f"factory.create() failed: {exc}")
        return {"ok": False, "error": "\\n".join(errors)}

    steps_ok = 0
    try:
        obs_reset = env.reset(seed=0)
        action = env.sample_action()
        step_out = env.step(action)
        render_out = env.render()
        env.close()
        steps_ok = 1
    except Exception as exc:
        errors.append(f"Episode run failed: {exc}\\n{traceback.format_exc()}")

    # Second short episode
    try:
        env2 = factory.create()
        env2.reset(seed=1)
        for _ in range(3):
            out = env2.step(env2.sample_action())
            terminated = getattr(out, "terminated", False)
            truncated = getattr(out, "truncated", False)
            if terminated or truncated:
                break
        env2.close()
        steps_ok += 1
    except Exception as exc:
        errors.append(f"Second episode failed: {exc}")

    return {
        "ok": len(errors) == 0,
        "steps_ok": steps_ok,
        "info_ok": info_ok,
        "errors": errors,
    }

result = _run(sys.argv[1], sys.argv[2])
print(json.dumps(result))
'''


def _safe_class_name(env_id: str) -> str:
    """Convert 'My-Env-v0' → 'MyEnv'."""
    import re
    parts = re.split(r"[-_\s/]+", env_id)
    # Drop trailing version token like 'v0', 'v1'
    if parts and re.fullmatch(r"v\d+", parts[-1], re.IGNORECASE):
        parts = parts[:-1]
    return "".join(p.capitalize() for p in parts if p)


def _load_module_from_source(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _strip_markdown_fences(code: str) -> str:
    """Remove ```python / ``` wrappers if present."""
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def rl_get_environment_template(
    env_id: str,
    description: str = "",
    tags: str = "custom",
    max_episode_steps: int = 200,
) -> str:
    """
    Return a Python source template for a new custom RLIP environment.

    The template is a complete, runnable skeleton that you can fill in.
    Once edited, pass it to ``rl_create_environment_from_code()`` to
    compile, validate, cache, and register the environment.

    Parameters
    ----------
    env_id:
        The identifier to register the environment under
        (e.g. ``"MyGame-v0"``).
    description:
        Short human-readable description (fills the ``EnvironmentInfo``).
    tags:
        Comma-separated tags (e.g. ``"custom,grid,discrete"``).
    max_episode_steps:
        Default episode step limit written into the template.

    Returns
    -------
    Python source code as a string ready to edit and pass back to
    ``rl_create_environment_from_code()``.
    """
    class_name = _safe_class_name(env_id) or "Custom"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    code = _TEMPLATE.format(
        env_id=env_id,
        class_name=class_name,
        description=description or f"Custom environment {env_id}.",
        tags_repr=repr(tag_list),
        max_episode_steps=max_episode_steps,
    )
    return (
        f"# Template for '{env_id}'  -  edit then pass to rl_create_environment_from_code()\n"
        f"# env_id='{env_id}'  class_prefix='{class_name}'\n\n"
        + code
    )


@mcp.tool()
def rl_validate_environment_code(
    env_id: str,
    python_code: str,
) -> str:
    """
    Validate custom environment Python source without saving or registering it.

    Compiles the code in an isolated subprocess, runs two short episodes,
    and reports any errors.  Use this to iterate on your implementation
    before calling ``rl_create_environment_from_code()``.

    Parameters
    ----------
    env_id:
        The environment ID expected to appear in the FACTORY's ``env_info``.
    python_code:
        Complete Python source defining a ``FACTORY`` instance (see
        ``rl_get_environment_template()`` for the required structure).

    Returns
    -------
    Validation report: success or a list of errors.
    """
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    code = _strip_markdown_fences(python_code)

    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "env_under_test.py"
        script_file = Path(tmp) / "validate.py"
        env_file.write_text(code, encoding="utf-8")
        script_file.write_text(_VALIDATION_SCRIPT, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(script_file), str(env_file), env_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "Validation timed out (30 s).  Check for infinite loops in reset() or step()."
        except Exception as exc:
            return f"Validation runner failed: {exc}"

        stderr = result.stderr.strip()
        stdout = result.stdout.strip()

        if result.returncode != 0:
            return (
                f"Environment code raised an exception during validation.\n\n"
                + (stderr or stdout or "(no output)")
            )

        try:
            report = json.loads(stdout)
        except Exception:
            return (
                f"Validation script produced unexpected output.\n"
                + (stderr or stdout or "(no output)")
            )

        if report.get("ok"):
            return (
                f"Validation passed for '{env_id}'.\n"
                f"  env_info:    {'ok' if report.get('info_ok') else 'failed'}\n"
                f"  episodes run: {report.get('steps_ok', 0)}/2\n\n"
                f"Call rl_create_environment_from_code(env_id='{env_id}', python_code=...) "
                f"to save and register."
            )
        else:
            errors = report.get("errors") or [report.get("error", "unknown error")]
            lines = [f"Validation failed for '{env_id}':\n"]
            for e in errors:
                lines.append(f"  • {e}")
            lines.append("\nFix the errors and call rl_validate_environment_code() again.")
            return "\n".join(lines)


@mcp.tool()
def rl_create_environment_from_code(
    env_id: str,
    python_code: str,
    description: str = "",
    tags: str = "custom",
    translator_code: str = "",
    validate: bool = True,
) -> str:
    """
    Create, validate, cache, and register a custom RLIP environment from
    Python source code.

    This is the primary tool for authoring entirely new RL environments.
    The workflow is:

    1. Call ``rl_get_environment_template(env_id=...)`` to get a skeleton.
    2. Fill in ``reset()``, ``step()``, ``render()``, and the spaces.
    3. Optionally call ``rl_validate_environment_code()`` to iterate.
    4. Call this tool to save and register the final version.

    The source is saved to ``~/.rlip/environments/<env_id>/env.py`` and
    a ``spec.json`` is written so ``rl_load_cached_environments()`` can
    restore it in future sessions.

    Parameters
    ----------
    env_id:
        The identifier to register the environment under
        (e.g. ``"MyGame-v0"``).  Must match the ``env_id`` inside the
        ``FACTORY.env_info`` returned by the code.
    python_code:
        Complete Python source.  Must define a module-level ``FACTORY``
        instance that is a ``RLIPEnvironmentFactory`` subclass (see the
        template from ``rl_get_environment_template()``).
    description:
        Human-readable description.  If empty, the value from
        ``FACTORY.env_info.description`` is used.
    tags:
        Comma-separated tags (e.g. ``"custom,discrete"``).  Merged with
        any tags already in ``FACTORY.env_info``.
    translator_code:
        Optional Python source for a ``translate(state, ...)`` function
        to attach as a language translator (same format as
        ``rl_set_translator_code()``).  Leave blank to add one later.
    validate:
        Run a quick validation episode before saving (default True).
        Set to False only if you are confident the code is correct.

    Returns
    -------
    Confirmation with the cache path and next-step suggestions.
    """
    import datetime as _dt  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..language_translation import TRANSLATORS  # noqa: PLC0415

    code = _strip_markdown_fences(python_code)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # ── 1. Optional validation ────────────────────────────────────────────────
    if validate:
        val_report = rl_validate_environment_code(env_id=env_id, python_code=code)
        if "Validation failed" in val_report or "exception" in val_report.lower():
            return (
                f"Aborting: environment code did not pass validation.\n\n"
                + val_report
                + "\n\nFix the errors and try again, or pass validate=False to skip."
            )

    # ── 2. Write source to cache ──────────────────────────────────────────────
    cache_root = _custom_env_cache_root()
    env_dir = cache_root / env_id
    env_dir.mkdir(parents=True, exist_ok=True)

    env_file = env_dir / "env.py"
    env_file.write_text(code, encoding="utf-8")

    # ── 3. Load the module and instantiate the factory ────────────────────────
    module_name = f"_rlip_custom_{env_id.replace('-', '_').replace('/', '_')}"
    try:
        module = _load_module_from_source(module_name, env_file)
    except Exception as exc:
        return f"Failed to import environment code: {exc}"

    from ..environments.base import RLIPEnvironmentFactory  # noqa: PLC0415

    factory = getattr(module, "FACTORY", None)
    if factory is None:
        for name in dir(module):
            obj = getattr(module, name)
            try:
                if isinstance(obj, RLIPEnvironmentFactory):
                    factory = obj
                    break
            except Exception:
                pass
    if factory is None:
        return (
            "No module-level FACTORY instance found in the provided code.\n"
            "Add: FACTORY = YourFactory()  at the bottom of the file."
        )

    # ── 4. Resolve metadata ───────────────────────────────────────────────────
    try:
        info = factory.env_info
    except Exception as exc:
        return f"factory.env_info raised an error: {exc}"

    final_description = description or info.description or f"Custom environment {env_id}."
    final_tags = list(dict.fromkeys(tag_list + list(info.tags or [])))

    # ── 5. Write spec.json ────────────────────────────────────────────────────
    spec: dict[str, Any] = {
        "env_id": env_id,
        "description": final_description,
        "tags": final_tags,
        "namespace": getattr(info, "namespace", "custom"),
        "max_episode_steps": getattr(info, "max_episode_steps", None),
        "render_modes": list(getattr(info, "render_modes", [])),
        "source": {
            "type": "class",
            "module_path": str(env_file),
            "class_name": "FACTORY",
            "init_kwargs": {},
        },
        "language_translation": {},
        "created_at": _dt.datetime.now().isoformat(),
        "updated_at": _dt.datetime.now().isoformat(),
    }

    # ── 6. Optional translator ────────────────────────────────────────────────
    translator_msg = ""
    if translator_code.strip():
        from ..language_translation.generator import (  # noqa: PLC0415
            GeneratedTranslator,
            _strip_markdown_fences as _sfences,
            _to_class_name,
        )

        tcode = _sfences(translator_code)

        def _no_llm(p: str) -> str:
            return ""

        try:
            gt = GeneratedTranslator(
                llm_fn=_no_llm,
                env_id=env_id,
                rule_code=tcode,
                env_context="",
                refine_threshold=99_999,
            )
            class_name = _to_class_name(env_id)
            t_path = env_dir / "translator.py"
            gt.save_code(t_path, class_name=class_name)
            _custom_translators[env_id] = gt
            TRANSLATORS[env_id] = gt
            spec["language_translation"] = {
                "type": "generated",
                "module_path": "translator.py",
                "class_name": class_name,
            }
            translator_msg = f"\n  Translator:       saved to {t_path}"
        except Exception as exc:
            translator_msg = f"\n  Translator:       FAILED - {exc}"

    (env_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # ── 7. Update catalog ─────────────────────────────────────────────────────
    catalog_path = _catalog_path()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, Any] = {"catalog_version": "0.2", "environments": []}
    if catalog_path.exists():
        try:
            catalog = json.loads(catalog_path.read_text())
        except Exception:
            pass
    envs: list[dict] = catalog.get("environments", [])
    updated = False
    for i, e in enumerate(envs):
        if e.get("env_id") == env_id:
            envs[i] = spec
            updated = True
            break
    if not updated:
        envs.append(spec)
    catalog["environments"] = envs
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    # ── 8. Register in this session ───────────────────────────────────────────
    try:
        _env_registry.register(factory)
    except Exception as exc:
        return f"Environment saved but registration failed: {exc}"

    _env_renders_dir(env_id).mkdir(parents=True, exist_ok=True)
    _env_agents_dir(env_id).mkdir(parents=True, exist_ok=True)
    _env_cache_dir(env_id).mkdir(parents=True, exist_ok=True)

    return (
        f"Environment '{env_id}' created and registered.\n\n"
        f"  Description:      {final_description}\n"
        f"  Tags:             {', '.join(final_tags) or '(none)'}\n"
        f"  Source:           {env_file}"
        + translator_msg
        + f"\n  Cache path:       {env_dir}\n\n"
        f"Next steps:\n"
        f"  • rl_sample_states_for_translation(env_id='{env_id}') - inspect raw states\n"
        f"  • rl_set_translator_code(env_id='{env_id}', ...) - add a language translator\n"
        f"  • rl_experiment_process(agent_type='tabular_q', env_id='{env_id}') - train an agent\n"
        f"  • rl_load_cached_environments() - restore this env in a future session"
    )


@mcp.tool()
def rl_delete_custom_environment(
    env_id: str,
    confirm: bool = False,
) -> str:
    """
    Delete a custom environment from the cache and catalog.

    This removes the environment's source files from
    ``~/.rlip/environments/<env_id>/`` and removes its entry from
    ``~/.rlip/catalog.json``.  The environment is also de-registered from
    the current session.

    Parameters
    ----------
    env_id:
        The environment ID to delete.
    confirm:
        Must be True to proceed.  Prevents accidental deletion.

    Returns
    -------
    Confirmation or error message.
    """
    import shutil  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    if not confirm:
        return (
            f"Pass confirm=True to delete '{env_id}'.\n"
            "This will remove all cached files and de-register the environment."
        )

    cache_dir = _custom_env_cache_root() / env_id
    removed_files = False
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            removed_files = True
        except Exception as exc:
            return f"Failed to remove cache directory {cache_dir}: {exc}"

    # Update catalog
    catalog_path = _catalog_path()
    catalog_updated = False
    if catalog_path.exists():
        try:
            catalog = json.loads(catalog_path.read_text())
            envs = [e for e in catalog.get("environments", []) if e.get("env_id") != env_id]
            catalog["environments"] = envs
            catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
            catalog_updated = True
        except Exception as exc:
            return f"Cache removed but catalog update failed: {exc}"

    # De-register from session
    try:
        _env_registry.deregister(env_id)
    except Exception:
        pass  # not all registry implementations support deregister

    # Remove translator
    _custom_translators.pop(env_id, None)
    try:
        from ..language_translation import TRANSLATORS  # noqa: PLC0415
        TRANSLATORS.pop(env_id, None)
    except Exception:
        pass

    parts = []
    if removed_files:
        parts.append(f"cache directory {cache_dir}")
    if catalog_updated:
        parts.append("catalog entry")
    return (
        f"Deleted '{env_id}': {', '.join(parts) or 'nothing found'}.\n"
        "The environment is no longer available in this session."
    )
