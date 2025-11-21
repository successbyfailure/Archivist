from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCHIVIST_")

    app_name: str = "Archivist RAG Service"
    default_namespace: str = "default"
    chunk_size: int = 800
    chunk_overlap: int = 100
    llm_backends: List[str] = Field(default_factory=lambda: ["openai", "anthropic", "local"])
    openai_api_key: str | None = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_text_model: str = "gpt-4o-mini"
    enable_vhs: bool = True
    metrics_window: int = 1000


@lru_cache()
def get_settings() -> Settings:
    return Settings()
