"""
RUNWISE FastAPI Application Entrypoint.
Conforms to PRD Section 7, Section 9, and Section 19.
Integrates routers from Person A (Graph/Experiments/Imports/Corrections)
and Person B (Recommendations/Copilot).
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Wire graph service into application state for Person B recommendation / copilot dependencies
app.state.graph_service = GraphRepository()

# CORS Middleware for frontend integration (Person C)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
