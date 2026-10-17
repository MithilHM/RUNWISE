"""
Configuration settings for RUNWISE backend application.
Supports environment-based configuration with sensible defaults.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RUNWISE"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ],
        description="Browser origins allowed to call the API. Comma separated in .env.",
    )

    # Optional LLM client (PRD 15: the app must work with no key at all)
    RUNWISE_LLM_API_KEY: Optional[str] = None
    RUNWISE_LLM_MODEL: str = "gpt-4o-mini"
    RUNWISE_LLM_BASE_URL: str = "https://api.openai.com/v1"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``A,B`` from the environment as well as a real list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

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
