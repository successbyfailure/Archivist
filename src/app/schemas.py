from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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


class CollectionSummary(BaseModel):
    namespace: str
    collection: str
    documents: int
    chunks: int
    latest_version: int
    languages: List[str] = Field(default_factory=list)
    latest_ingested_at: Optional[datetime] = None


class DocumentPreview(BaseModel):
    id: str
    version: int
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    chunks: int
    snippet: str
    language: Optional[str] = None
