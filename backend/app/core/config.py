"""
Configuration settings for RUNWISE backend application.
Supports environment-based configuration with sensible defaults.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RUNWISE"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # FalkorDB Connection Settings
    FALKORDB_HOST: str = Field(default="127.0.0.1", description="FalkorDB host address")
    FALKORDB_PORT: int = Field(default=6379, description="FalkorDB port")
    FALKORDB_GRAPH: str = Field(default="runwise", description="FalkorDB default graph name")
    FALKORDB_PASSWORD: Optional[str] = Field(default=None, description="FalkorDB password if configured")
    FALKORDB_SSL: bool = Field(default=False, description="Use SSL for FalkorDB connection")
    FALKORDB_TIMEOUT_MS: int = Field(default=5000, description="Query timeout in milliseconds")
    FALKORDB_MAX_CONNECTIONS: int = Field(default=20, description="Max connection pool size")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
