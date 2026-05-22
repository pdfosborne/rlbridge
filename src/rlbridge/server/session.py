"""
Session Manager
================
Tracks live environment instances across requests.
Each instance is identified by a UUID string.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..environments.base import rlbridgeEnvironment
from ..protocol.constants import ErrorCodes
from .exceptions import rlbridgeError


@dataclass
class InstanceRecord:
    instance_id: str
    env_id: str
    environment: rlbridgeEnvironment
    render_mode: Optional[str]
    created_kwargs: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Thread-safe store of active environment instances."""

    def __init__(self, max_instances: int = 64) -> None:
        self._instances: dict[str, InstanceRecord] = {}
        self._lock = threading.RLock()
        self.max_instances = max_instances

    def create_instance(
        self,
        env_id: str,
        environment: rlbridgeEnvironment,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> InstanceRecord:
        with self._lock:
            if len(self._instances) >= self.max_instances:
                raise rlbridgeError(
                    code=ErrorCodes.ENV_CREATION_FAILED,
                    message=f"Maximum instance limit ({self.max_instances}) reached. "
                            "Close unused environments first.",
                )
            instance_id = str(uuid.uuid4())
            record = InstanceRecord(
                instance_id=instance_id,
                env_id=env_id,
                environment=environment,
                render_mode=render_mode,
                created_kwargs=dict(kwargs),
            )
            self._instances[instance_id] = record
            return record

    def get_instance(self, instance_id: str) -> InstanceRecord:
        with self._lock:
            record = self._instances.get(instance_id)
        if record is None:
            raise rlbridgeError(
                code=ErrorCodes.INSTANCE_NOT_FOUND,
                message=f"No live instance with id '{instance_id}'",
                data={"instance_id": instance_id},
            )
        return record

    def close_instance(self, instance_id: str) -> bool:
        with self._lock:
            record = self._instances.pop(instance_id, None)
        if record is None:
            return False
        record.environment.close()
        return True

    def list_instances(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._instances.values())
        return [
            {
                "instance_id": r.instance_id,
                "env_id": r.env_id,
                "render_mode": r.render_mode,
                "is_initialized": r.environment.is_initialized,
            }
            for r in records
        ]

    def close_all(self) -> int:
        with self._lock:
            ids = list(self._instances.keys())
        count = 0
        for iid in ids:
            if self.close_instance(iid):
                count += 1
        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._instances)
