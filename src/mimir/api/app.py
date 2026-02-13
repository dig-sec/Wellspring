from __future__ import annotations

import ipaddress
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from .routes import router

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def _is_loopback_host(host: str) -> bool:
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host.split("%", 1)[0])
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            return addr.ipv4_mapped.is_loopback
        return addr.is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _request_token(request: Request) -> str:
    header_value = request.headers.get("authorization", "")
    if header_value.lower().startswith("bearer "):
        return header_value[7:].strip()
    return request.headers.get("x-api-key", "").strip()


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
    path = request.url.path
    if request.method == "OPTIONS" or path.startswith("/static"):
        return await call_next(request)

    configured_token = settings.api_token.strip()
    if configured_token:
        provided = _request_token(request)
        if not provided or not secrets.compare_digest(provided, configured_token):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    if settings.allow_localhost_without_token:
        client_host = request.client.host if request.client else ""
        if _is_loopback_host(client_host):
            return await call_next(request)

    logger.warning(
        "Blocked unauthenticated request from host=%s path=%s",
        request.client.host if request.client else "",
        path,
    )
    return JSONResponse(
        status_code=403,
        content={
            "detail": (
                "Access denied. Configure MIMIR_API_TOKEN or enable "
                "MIMIR_ALLOW_LOCALHOST_WITHOUT_TOKEN for local use."
            )
        },
    )


# Serve static assets (CSS, JS)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

app.include_router(router)
