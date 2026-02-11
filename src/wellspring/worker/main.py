from __future__ import annotations

import asyncio
import logging

from ..config import get_settings
from ..pipeline.runner import process_run
from ..storage.sqlite_store import SQLiteGraphStore, SQLiteRunStore

logger = logging.getLogger(__name__)


async def worker_loop() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    graph_store = SQLiteGraphStore(settings.db_path)
    run_store = SQLiteRunStore(settings.db_path)

    recovered = run_store.recover_stale_runs()
    if recovered:
        logger.info("Recovered %d stale run(s) back to pending", recovered)

    while True:
        run = run_store.claim_next_run()
        if not run:
            await asyncio.sleep(2)
            continue
        logger.info("Processing run %s", run.run_id)
        try:
            await process_run(run.run_id, graph_store, run_store, settings)
            run_store.update_run_status(run.run_id, "completed")
            logger.info("Run %s completed", run.run_id)
        except Exception as exc:
            run_store.update_run_status(run.run_id, "failed", error=str(exc))
            logger.exception("Run %s failed", run.run_id)


def main() -> None:
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
