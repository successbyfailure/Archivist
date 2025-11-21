from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    collection: str
    namespace: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class URLIngestionRequest(IngestionRequest):
    url: str


class GitIngestionRequest(IngestionRequest):
    repo_url: str
    branch: Optional[str] = None
    path: Optional[str] = None


class APIIngestionRequest(IngestionRequest):
    endpoint: str
    headers: Dict[str, str] = Field(default_factory=dict)


class WebhookEvent(IngestionRequest):
    payload: Dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    collection: Optional[str] = None
    filters: Dict[str, str] = Field(default_factory=dict)
    pipeline: Optional[str] = None
    retrieval_only: bool = False
    stream: bool = False
    structured: bool = False
    partial: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(QueryRequest):
    history: List[ChatMessage] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    name: str
    description: str
    optimized: bool = False
    parameters: Dict[str, str] = Field(default_factory=dict)


class ReplayRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    collection: Optional[str] = None
