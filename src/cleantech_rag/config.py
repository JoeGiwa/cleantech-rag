"""Application settings, loaded from environment variables (and an optional local dotenv file)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chunks fetched per wanted article; single source for Settings and the retrieval tools.
DEFAULT_RETRIEVAL_OVERFETCH = 4


# USD per 1M tokens as (input, output). Approximate list prices; override via MODEL_PRICES (JSON).
DEFAULT_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),  # query embeddings (input only)
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets (optional so import/tests never crash without them)
    openai_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    # Models
    agent_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    agent_recursion_limit: int = Field(12, ge=1)  # max LangGraph steps per question

    max_history: int = Field(
        10, ge=1
    )  # previous turns kept per thread (the current turn is always included)

    # Price table for eval cost estimates, matched to a response's model name by longest prefix
    # (so "gpt-4o-mini-2024-07-18" uses "gpt-4o-mini"). Env: MODEL_PRICES='{"gpt-4o":[2.5,10]}'
    model_prices: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_MODEL_PRICES)
    )

    # Chunking / retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    retrieval_overfetch: int = DEFAULT_RETRIEVAL_OVERFETCH  # chunks fetched per wanted article

    # Paths
    raw_data_dir: Path = Path("data/raw")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "cleantech_articles"

    # Brave web search
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    brave_timeout: float = 10.0
    brave_result_count: int = 5


def get_settings() -> Settings:
    return Settings()
