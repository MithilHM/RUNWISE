"""
RUNWISE FastAPI Application Entrypoint.
Conforms to PRD Section 7, Section 9, and Section 19.
Integrates the graph-layer routers (Graph/Experiments/Imports/Corrections)
and the reasoning-layer routers (Recommendations/Copilot).
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.graph.client import graph_client
from app.graph.schema import initialize_schema
from app.graph.repository import GraphRepository
from app.api.projects import router as projects_router
from app.api.imports import router as imports_router
from app.api.experiments import router as experiments_router
from app.api.corrections import router as corrections_router
from app.api.recommendations import router as recommendations_router
from app.api.copilot import router as copilot_router
from app.api.runtime import router as runtime_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("runwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: connects to FalkorDB and ensures schema indices exist."""
    logger.info("Initializing RUNWISE backend application...")
    try:
        health = graph_client.check_health()
        if health.get("status") == "healthy":
            logger.info(f"FalkorDB is online at {settings.FALKORDB_HOST}:{settings.FALKORDB_PORT}. Initializing schema...")
            graph = graph_client.get_graph()
            res = initialize_schema(graph)
            logger.info(f"Graph schema initialized: {res}")
        else:
            logger.warning(f"FalkorDB health check indicates: {health}. Application starting in decoupled mode.")
    except Exception as e:
        logger.warning(f"Failed to initialize FalkorDB on startup: {e}. Graph operations will reconnect on demand.")

    yield

    logger.info("Shutting down RUNWISE backend application...")
    graph_client.close()


app = FastAPI(
    title=settings.APP_NAME,
    description="RUNWISE — AI Research Experiment Copilot Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# Wire the graph service into application state for the recommendation / copilot dependencies
app.state.graph_service = GraphRepository()

# CORS for the frontend dev server. Origins are explicit: ``allow_origins=["*"]``
# combined with credentials is rejected by browsers, and an open origin on a
# service with no authentication is a hole rather than a convenience.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# One error envelope for every failure (PRD 9: structured errors everywhere)
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
        content={
            "detail": _as_detail_list(
                exc.detail, code_by_status.get(exc.status_code, "http_error")
            )
        },
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

    The frontend can then show "the graph is unavailable, retry" instead of a parse
    error, and the message never leaks a driver stack trace to a client.
    """
    logger.exception("Unhandled error serving %s %s", request.method, request.url.path)
    if exc.__class__.__module__.startswith("falkordb"):
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
    """Health check endpoint providing FalkorDB status."""
    db_health = graph_client.check_health()
    return {
        "status": "healthy" if db_health.get("status") == "healthy" else "degraded",
        "app": settings.APP_NAME,
        "database": db_health
    }


# Router Registration Points (PRD Section 9 & 19)
app.include_router(projects_router, prefix=settings.API_V1_STR)
app.include_router(imports_router, prefix=settings.API_V1_STR)
app.include_router(experiments_router, prefix=settings.API_V1_STR)
app.include_router(corrections_router, prefix=settings.API_V1_STR)
app.include_router(recommendations_router, prefix=settings.API_V1_STR)
app.include_router(copilot_router, prefix=settings.API_V1_STR)
app.include_router(runtime_router, prefix="/api")
app.include_router(runtime_router, prefix=settings.API_V1_STR)
