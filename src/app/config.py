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
    enable_vhs: bool = True
    metrics_window: int = 1000


@lru_cache()
def get_settings() -> Settings:
    return Settings()
