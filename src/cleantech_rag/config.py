"""Application settings, loaded from environment variables (and an optional local dotenv file)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    max_history: int = Field(10, ge=1)  # conversation turns kept per thread

    # Chunking / retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    retrieval_overfetch: int = 4  # chunks fetched per wanted article

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
