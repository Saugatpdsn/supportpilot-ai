"""FastAPI application: wiring, middleware, and exception handlers. No business logic here."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import router
from app.config import ConfigError, get_settings
from app.guardrails import RedactingFilter
from app.rag import ensure_ingested

logger = logging.getLogger("supportpilot")
MAX_BODY_BYTES = 10_000


def configure_logging() -> None:
    logging.basicConfig(level=get_settings().log_level)
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        ensure_ingested()  # builds the ChromaDB collection on first start
    except ConfigError:
        logger.warning("GOOGLE_API_KEY is not set: the knowledge base was not loaded.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Knowledge base is not ready: %s", type(exc).__name__)
    yield


app = FastAPI(
    title="SupportPilot AI",
    version="0.1.0",
    description="Agentic customer support assistant (LangGraph + Gemini) with human-in-the-loop approval.",
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large."})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Only field + message: never echo the submitted input back.
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "invalid value")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError):
    logger.error("Service is not configured")
    return JSONResponse(
        status_code=503,
        content={"detail": "The service is not configured. Please contact the administrator."},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = uuid.uuid4().hex[:12]
    logger.error("Unhandled %s (request_id=%s)", type(exc).__name__, request_id)  # no message/traceback
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error.", "request_id": request_id}
    )