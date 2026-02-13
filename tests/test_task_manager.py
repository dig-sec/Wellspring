from __future__ import annotations

import asyncio

from mimir.api.tasks import TaskManager, TaskStatus


def test_task_manager_prunes_finished_history():
    manager = TaskManager(max_history=50)
    created = [manager.create("filesystem_scan") for _ in range(55)]
    for task in created:
        manager.update(task.id, status=TaskStatus.COMPLETED)

    newest = manager.create("filesystem_scan")
    remaining = manager.list_all()

    assert len(remaining) == 50
    assert newest.id in {task.id for task in remaining}


def test_task_manager_marks_async_failures():
    async def _exercise() -> None:
        manager = TaskManager(max_history=50)
        task = manager.create("opencti_pull")
        manager.update(task.id, status=TaskStatus.RUNNING)

        async def _boom():
            raise RuntimeError("boom")

        manager.start_async(task.id, _boom())

        for _ in range(40):
            current = manager.get(task.id)
            if current and current.status == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.01)

        current = manager.get(task.id)
        assert current is not None
        assert current.status == TaskStatus.FAILED
        assert current.finished_at
        assert "boom" in (current.error or "")

    asyncio.run(_exercise())
