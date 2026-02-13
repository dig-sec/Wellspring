"""Background task manager for long-running operations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    id: str
    kind: str  # "opencti_pull", "filesystem_scan", etc.
    status: TaskStatus = TaskStatus.PENDING
    progress: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class TaskManager:
    """Simple in-memory task tracker for background jobs."""

    def __init__(self, max_history: int = 500):
        self._tasks: Dict[str, BackgroundTask] = {}
        self._running: Dict[str, asyncio.Task] = {}
        self._max_history = max(50, max_history)

    def _prune(self) -> None:
        # Drop completed coroutine handles.
        for task_id, task in list(self._running.items()):
            if task.done():
                self._running.pop(task_id, None)

        # Keep bounded history for finished tasks.
        if len(self._tasks) <= self._max_history:
            return

        removable = [
            t
            for t in self._tasks.values()
            if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
        ]
        removable.sort(key=lambda t: t.started_at or "")
        while len(self._tasks) > self._max_history and removable:
            victim = removable.pop(0)
            self._tasks.pop(victim.id, None)

    def create(self, kind: str, detail: Optional[Dict] = None) -> BackgroundTask:
        self._prune()
        task = BackgroundTask(
            id=str(uuid4())[:8],
            kind=kind,
            status=TaskStatus.PENDING,
            detail=detail or {},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Optional[BackgroundTask]:
        return self._tasks.get(task_id)

    def list_all(self) -> List[BackgroundTask]:
        self._prune()
        return list(self._tasks.values())

    def update(self, task_id: str, **kwargs):
        t = self._tasks.get(task_id)
        if t:
            for k, v in kwargs.items():
                setattr(t, k, v)

    def start_async(self, task_id: str, coro):
        """Kick off a coroutine and track it."""
        async_task = asyncio.create_task(coro)
        self._running[task_id] = async_task

        async def _wrapper():
            try:
                await async_task
            except Exception as exc:
                logger.exception("Background task %s failed", task_id)
                self.update(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
            finally:
                self._running.pop(task_id, None)
                self._prune()

        asyncio.create_task(_wrapper())


# Singleton
task_manager = TaskManager()
