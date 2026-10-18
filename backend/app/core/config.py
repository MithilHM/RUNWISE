"""Configuration for the RUNWISE ML harness."""

from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "RUNWISE"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    CORS_ORIGINS: Union[List[str], str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ],
        description="Browser origins allowed to call the API. Comma separated or JSON array.",
    )

    # Uploads -----------------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 64 * 1024 * 1024
    MAX_UPLOAD_ROWS: int = 2_000_000
    DATA_DIR: str = "runs_data"

    # Harness -----------------------------------------------------------------
    DEFAULT_TRIAL_BUDGET: int = 12
    MAX_TRIAL_BUDGET: int = 200
    TRAIN_TEST_SPLIT: float = 0.2
    RANDOM_SEED: int = 42

    # Laya --------------------------------------------------------------------
    # Leave RUNWISE_LLM_API_KEY unset and Laya runs its deterministic graph policy.
    # The response always reports which one produced a decision.
    RUNWISE_LLM_API_KEY: Optional[str] = None
    RUNWISE_LLM_MODEL: str = "gpt-4o-mini"
    RUNWISE_LLM_BASE_URL: str = "https://api.openai.com/v1"
    LAYA_TIMEOUT_SECONDS: float = 20.0

    # FalkorDB ----------------------------------------------------------------
    FALKORDB_HOST: str = Field(default="127.0.0.1")
    FALKORDB_PORT: int = Field(default=6379)
    FALKORDB_GRAPH: str = Field(default="runwise")
    FALKORDB_PASSWORD: Optional[str] = Field(default=None)
    FALKORDB_SSL: bool = Field(default=False)

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> List[str]:
        """Accept a comma-separated string, a JSON array string, or a list."""
        if isinstance(value, str):
            val = value.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    import json

                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in val.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    model_config = SettingsConfigDict(
        # `.env` lives at the repository root, and the documented way to start the
        # server is `cd backend && uvicorn app.main:app`, so looking only at the
        # working directory silently finds nothing. Both locations are searched.
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()