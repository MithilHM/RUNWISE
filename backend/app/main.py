"""RUNWISE FastAPI application.

An autonomous ML harness: upload one static CSV, choose from three models, watch
the dependency graph fill in, and ask why any of it happened.

The startup sequence matters more than it looks. `write_models` must run before
the first `propose`, because Laya grounds every decision in `:Model` and `:Param`
nodes and a highlight pointing at a node that was never created shows the user
nothing.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    catalog_router,
    datasets_router,
    graph_router,
    mcp_router,
    runs_router,
)
from app.core.config import settings
from app.graph.client import graph_client
from app.graph.repository import get_graph_store
from app.services.catalog import list_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("runwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connects to FalkorDB, ensures the schema, and writes the model catalog.

    FalkorDB is required, not optional. The graph *is* the product, so a missing
    database is a loud degraded start rather than a silent fallback to a local
    store that would make the tool claim something untrue.
    """
    logger.info("Starting %s...", settings.APP_NAME)
    try:
        health = graph_client.check_health()
        if health.get("status") == "healthy":
            store = get_graph_store()
            store.ensure_schema()
            params = store.write_models(list_models())
            logger.info(
                "FalkorDB online at %s:%s. Schema ready, %d parameter node(s) written.",
                settings.FALKORDB_HOST, settings.FALKORDB_PORT, params,
            )
        else:
            logger.warning(
                "FalkorDB health check reports %s. The app will start but graph "
                "operations will fail until it is reachable.",
                health.get("status"),
            )
    except Exception as exc:  # noqa: BLE001 - startup must report, not crash
        logger.error(
            "Could not initialise FalkorDB: %s. Endpoints needing the graph will "
            "return 503 until it is reachable.", exc,
        )

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)
    graph_client.close()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "An autonomous ML harness. Upload a CSV, pick from RandomForest, XGBoost "
        "or LightGBM, and watch every hyperparameter value, trial, decision and "
        "log line land in the graph — then ask why."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# One error envelope for every failure
# ---------------------------------------------------------------------------


def _as_detail_list(detail: object, fallback_code: str) -> list[dict]:
    """Coerce any exception detail into the canonical list of error objects."""
    if isinstance(detail, list) and detail and all(isinstance(d, dict) and "code" in d for d in detail):
        return [
            {"code": str(d.get("code")), "message": str(d.get("message", "")), "field": d.get("field")}
            for d in detail
        ]
    if isinstance(detail, dict) and "code" in detail:
        return [{"code": str(detail["code"]), "message": str(detail.get("message", "")), "field": detail.get("field")}]
    return [{"code": fallback_code, "message": str(detail), "field": None}]


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code_by_status = {
        status.HTTP_400_BAD_REQUEST: "bad_request",
        status.HTTP_401_UNAUTHORIZED: "unauthorized",
        status.HTTP_403_FORBIDDEN: "forbidden",
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_409_CONFLICT: "conflict",
        status.HTTP_422_UNPROCESSABLE_ENTITY: "validation_error",
        status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": _as_detail_list(exc.detail, code_by_status.get(exc.status_code, "http_error"))},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Report a body or query problem in the same shape as every other error."""
    errors = [
        {
            "code": "validation_error",
            "message": f"{'.'.join(str(p) for p in err.get('loc', [])[1:])}: {err.get('msg', 'invalid value')}",
            "field": ".".join(str(p) for p in err.get("loc", [])[1:]) or None,
        }
        for err in exc.errors()
    ]
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """A database failure is a 503 with a code, not an opaque 500 with a traceback.

    The frontend can then say "the graph is unavailable, retry" instead of
    rendering a parse error, and the message never leaks a driver stack trace.
    """
    logger.exception("Unhandled error serving %s %s", request.method, request.url.path)
    if exc.__class__.__module__.startswith(("falkordb", "redis")):
        code, http_status, message = (
            "graph_unavailable",
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The graph database is not reachable. Retry once FalkorDB is back.",
        )
    else:
        code, http_status, message = (
            "internal_error",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "The request could not be completed.",
        )
    return JSONResponse(status_code=http_status, content={"detail": [{"code": code, "message": message, "field": None}]})


@app.get("/health", tags=["system"])
def health_check():
    """Liveness plus the FalkorDB status the console shows in its header."""
    db_health = graph_client.check_health()
    return {
        "status": "healthy" if db_health.get("status") == "healthy" else "degraded",
        "app": settings.APP_NAME,
        "database": db_health,
    }


app.include_router(datasets_router, prefix=settings.API_V1_STR)
app.include_router(catalog_router, prefix=settings.API_V1_STR)
app.include_router(runs_router, prefix=settings.API_V1_STR)
app.include_router(graph_router, prefix=settings.API_V1_STR)
app.include_router(mcp_router, prefix=settings.API_V1_STR)