"""
FalkorDB Database Client for RUNWISE.
Provides connection pooling, health checks, safe parameterized queries,
isolated error handling, and graph selection.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from falkordb import FalkorDB
from falkordb.graph import Graph

from app.core.config import settings

logger = logging.getLogger(__name__)


class FalkorDBError(Exception):
    """Base exception for all FalkorDB-related errors."""
    pass


class FalkorDBConnectionError(FalkorDBError):
    """Raised when connecting or communicating with FalkorDB fails."""
    pass


class FalkorDBQueryError(FalkorDBError):
    """Raised when a Cypher query execution fails."""
    pass


class FalkorDBClient:
    """
    Manages connection to FalkorDB and provides safe execution wrappers.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        graph_name: Optional[str] = None,
        password: Optional[str] = None,
        ssl: Optional[bool] = None,
    ):
        self.host = host or settings.FALKORDB_HOST
        self.port = port or settings.FALKORDB_PORT
        self.default_graph_name = graph_name or settings.FALKORDB_GRAPH
        self.password = password or settings.FALKORDB_PASSWORD
        self.ssl = ssl if ssl is not None else settings.FALKORDB_SSL

        self._driver: Optional[FalkorDB] = None
        self._cached_graphs: Dict[str, Graph] = {}

    def connect(self) -> FalkorDB:
        """Establish or return the active FalkorDB driver."""
        if self._driver is None:
            try:
                self._driver = FalkorDB(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    ssl=self.ssl
                )
                logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to FalkorDB at {self.host}:{self.port}: {e}")
                raise FalkorDBConnectionError(f"Could not connect to FalkorDB at {self.host}:{self.port}: {e}")
        return self._driver

    def get_graph(self, graph_name: Optional[str] = None) -> Graph:
        """Retrieve a specific graph handle from the driver."""
        name = graph_name or self.default_graph_name
        if name not in self._cached_graphs:
            driver = self.connect()
            try:
                self._cached_graphs[name] = driver.select_graph(name)
            except Exception as e:
                raise FalkorDBConnectionError(f"Failed to select graph '{name}': {e}")
        return self._cached_graphs[name]

    def check_health(self) -> Dict[str, Any]:
        """Verify database connectivity and responsiveness."""
        try:
            start_t = time.perf_counter()
            graph = self.get_graph()
            res = graph.query("RETURN 1 AS ping")
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            is_healthy = bool(res and res.result_set and res.result_set[0][0] == 1)
            return {
                "status": "healthy" if is_healthy else "degraded",
                "host": self.host,
                "port": self.port,
                "graph": self.default_graph_name,
                "latency_ms": round(latency_ms, 2)
            }
        except Exception as e:
            logger.warning(f"FalkorDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "host": self.host,
                "port": self.port,
                "graph": self.default_graph_name,
                "error": str(e)
            }

    def execute_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        graph_name: Optional[str] = None,
        read_only: bool = False
    ) -> List[List[Any]]:
        """
        Execute a parameterized Cypher query safely.
        Returns the raw result set (rows of values).
        """
        graph = self.get_graph(graph_name)
        start_time = time.perf_counter()
        try:
            query_params = params or {}
            # FalkorDB python client supports params dict
            res = graph.query(cypher, params=query_params)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"Executed Cypher in {duration_ms:.2f}ms: {cypher[:100]}")
            return res.result_set if res and hasattr(res, "result_set") else []
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Cypher execution failed in {duration_ms:.2f}ms: {e}\nQuery: {cypher}\nParams: {params}")
            raise FalkorDBQueryError(f"Error executing Cypher query: {e}")

    def delete_graph(self, graph_name: Optional[str] = None) -> bool:
        """Deletes all nodes and relationships in a graph (useful for tests and resets)."""
        name = graph_name or self.default_graph_name
        try:
            graph = self.get_graph(name)
            graph.delete()
            if name in self._cached_graphs:
                del self._cached_graphs[name]
            return True
        except Exception as e:
            logger.warning(f"Error deleting graph {name}: {e}")
            return False

    def close(self):
        """Close driver connections."""
        self._cached_graphs.clear()
        self._driver = None


# Singleton instance for general backend application use
graph_client = FalkorDBClient()


def get_graph_client() -> FalkorDBClient:
    """Dependency provider for FastAPI or standalone services."""
    return graph_client
