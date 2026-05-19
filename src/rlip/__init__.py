"""
RLIP – Reinforcement Learning Interaction Protocol
====================================================
A JSON-RPC 2.0-based protocol for connecting AI agents to RL environments.
Surfaces as a Claude Code MCP plugin, and as an optional standalone HTTP server.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rlip")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from .interaction_protocols import (
    InteractionResult,
    EpisodeResult,
    StepRecord,
    SingleStepProtocol,
    RandomEpisodeProtocol,
    GreedyEpisodeProtocol,
    MultiEpisodeProtocol,
    InstructionFollowingProtocol,
    list_protocols,
    run_protocol,
)

from .instruction_following import (
    TextEncoder,
    TFIDFEncoder,
    InstructionMatch,
    InstructionCacheEntry,
    match_instruction,
    build_sequential_instruction_following_protocol,
    infer_max_reward,
    scale_sub_goal_bonus,
    LanguageTrackingWrapper,
    derive_instructions_from_training,
    train_and_derive_instructions,
    list_instruction_cache,
    clear_instruction_cache,
    state_instruction_map,
    clear_obs_cache,
    obs_cache_info,
    obs_cache_langs,
)

from .instruction_matching import (
    BaseEncoder,
    BM25Encoder,
    SentenceEncoder,
    get_encoder,
)

from .language_translation import (
    LanguageTranslator,
    SailingLanguageTranslator,
    CachingTranslator,
    get_translator,
    translate,
    clear_translation_cache,
    translation_cache_info,
    LLMCallable,
    GeneratedTranslator,
    TranslatorGenerator,
    build_translator,
)

from .policy_rendering import (
    extract_optimal_policy,
    RenderedFrame,
    PolicyRenderResult,
    PolicyRenderer,
    save_frames_to_dir,
    save_gif,
    render_optimal_policy,
)

from .environments.builder import (
    EnvSpec,
    BuiltEnvironment,
    EnvironmentBuilder,
    load_cached_environments,
)

try:
    from .rl_agents import (
        TabularQAgent,
        TabularQTrainResult,
        DQNAgent,
        DQNTrainResult,
        PPOAgent,
        PPOTrainResult,
        LocalLLMAgent,
    )
    _RL_AGENTS_AVAILABLE = True
except ImportError:
    _RL_AGENTS_AVAILABLE = False

__all__ = [
    "__version__",
    # Interaction protocols
    "InteractionResult",
    "EpisodeResult",
    "StepRecord",
    "SingleStepProtocol",
    "RandomEpisodeProtocol",
    "GreedyEpisodeProtocol",
    "MultiEpisodeProtocol",
    "InstructionFollowingProtocol",
    "list_protocols",
    "run_protocol",
    # Instruction following
    "TextEncoder",
    "TFIDFEncoder",
    "InstructionMatch",
    "InstructionCacheEntry",
    "match_instruction",
    "build_sequential_instruction_following_protocol",
    "infer_max_reward",
    "scale_sub_goal_bonus",
    "LanguageTrackingWrapper",
    "derive_instructions_from_training",
    "train_and_derive_instructions",
    "list_instruction_cache",
    "clear_instruction_cache",
    "state_instruction_map",
    "clear_obs_cache",
    "obs_cache_info",
    "obs_cache_langs",
    # Instruction matching encoders
    "BaseEncoder",
    "BM25Encoder",
    "SentenceEncoder",
    "get_encoder",
    # Language translation
    "LanguageTranslator",
    "SailingLanguageTranslator",
    "CachingTranslator",
    "get_translator",
    "translate",
    "clear_translation_cache",
    "translation_cache_info",
    "LLMCallable",
    "GeneratedTranslator",
    "TranslatorGenerator",
    "build_translator",
    # Policy rendering
    "extract_optimal_policy",
    "RenderedFrame",
    "PolicyRenderResult",
    "PolicyRenderer",
    "save_frames_to_dir",
    "save_gif",
    "render_optimal_policy",
    # Environment builder
    "EnvSpec",
    "BuiltEnvironment",
    "EnvironmentBuilder",
    "load_cached_environments",
]

if _RL_AGENTS_AVAILABLE:
    __all__.extend([
        "TabularQAgent",
        "TabularQTrainResult",
        "DQNAgent",
        "DQNTrainResult",
        "PPOAgent",
        "PPOTrainResult",
        "LocalLLMAgent",
    ])
