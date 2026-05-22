"""RLIP application-level exception."""

from __future__ import annotations

from typing import Any, Optional


class RLIPError(Exception):
    """Raised by RLIP server components; maps directly to a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err
