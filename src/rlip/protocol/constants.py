"""RLIP protocol constants."""

RLIP_PROTOCOL_VERSION = "0.1"
JSONRPC_VERSION = "2.0"

# ── Method names ────────────────────────────────────────────────────────────
class Methods:
    INITIALIZE          = "rlip/initialize"
    LIST_ENVIRONMENTS   = "rlip/environments/list"
    CREATE_ENVIRONMENT  = "rlip/environment/create"
    RESET               = "rlip/environment/reset"
    STEP                = "rlip/environment/step"
    GET_SPACES          = "rlip/environment/spaces"
    RENDER              = "rlip/environment/render"
    CLOSE               = "rlip/environment/close"
    LIST_INSTANCES      = "rlip/instances/list"

# ── JSON-RPC error codes (–32000 to –32099 are application errors) ──────────
class ErrorCodes:
    # JSON-RPC standard
    PARSE_ERROR      = -32700
    INVALID_REQUEST  = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS   = -32602
    INTERNAL_ERROR   = -32603

    # RLIP application errors
    ENV_NOT_FOUND        = -32000
    INSTANCE_NOT_FOUND   = -32001
    ENV_NOT_INITIALIZED  = -32002
    INVALID_ACTION       = -32003
    RENDER_UNAVAILABLE   = -32004
    ENV_CREATION_FAILED  = -32005
    PROTOCOL_MISMATCH    = -32006

ERROR_MESSAGES: dict[int, str] = {
    ErrorCodes.ENV_NOT_FOUND:       "Environment ID not found in registry",
    ErrorCodes.INSTANCE_NOT_FOUND:  "Environment instance not found (was it closed?)",
    ErrorCodes.ENV_NOT_INITIALIZED: "Environment must be reset before stepping",
    ErrorCodes.INVALID_ACTION:      "Action is incompatible with the environment's action space",
    ErrorCodes.RENDER_UNAVAILABLE:  "Rendering is not supported by this environment",
    ErrorCodes.ENV_CREATION_FAILED: "Failed to create environment instance",
    ErrorCodes.PROTOCOL_MISMATCH:   "Client/server RLIP protocol version mismatch",
}
