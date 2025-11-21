from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    embedding: List[float]
    score: float = 0.0
    metadata: Dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    id: str
    namespace: str
    collection: str
    version: int
    content: str
    chunks: List[Chunk]
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    language: Optional[str] = None


class QueryResult(BaseModel):
    answer: str
    chunks: List[Chunk]
    debug_info: Dict[str, str] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    documents: int
    chunks: int
    vector_size: int
    avg_chunk_length: float
    latency_ms: float
    tokens_used: int
    latest_queries: List[Dict[str, str]] = Field(default_factory=list)
