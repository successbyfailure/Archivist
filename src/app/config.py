from functools import lru_cache
from pydantic import BaseSettings, Field
from typing import List


class Settings(BaseSettings):
    app_name: str = "Archivist RAG Service"
    default_namespace: str = "default"
    chunk_size: int = 800
    chunk_overlap: int = 100
    llm_backends: List[str] = Field(default_factory=lambda: ["openai", "anthropic", "local"])
    enable_vhs: bool = True
    metrics_window: int = 1000

    class Config:
        env_prefix = "ARCHIVIST_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
