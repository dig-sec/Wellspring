from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from .access import authorize_request
from .routes import router

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connector sync and LLM extraction are now handled by dedicated
    # worker processes (feedly_worker, opencti_worker, elastic_worker,
    # llm_worker).  The API process no longer runs an inline scheduler.
    #
    # Manual sync endpoints in routes.py still work — they just trigger
    # one-shot syncs on demand.
    logging.getLogger(__name__).info(
        "API started.  Data ingestion handled by separate worker processes."
    )
    yield


app = FastAPI(title="Mimir API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def enforce_access_controls(request: Request, call_next):
    allowed, status_code, detail = authorize_request(
        request,
        api_token=settings.api_token,
        allow_localhost_without_token=settings.allow_localhost_without_token,
    )
    if allowed:
        return await call_next(request)

    if status_code == 403:
        logger.warning(
            "Blocked unauthenticated request from host=%s path=%s",
            request.client.host if request.client else "",
            request.url.path,
        )
    return JSONResponse(status_code=status_code, content={"detail": detail})


# Serve static assets (CSS, JS)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

app.include_router(router)
