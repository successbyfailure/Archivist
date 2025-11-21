import asyncio
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .ingestion.pipelines import IngestionPipeline
from .metrics import MetricsRecorder
from .pipelines.dspy import PipelineRegistry
from .schemas import (
    ChatRequest,
    CollectionSummary,
    DocumentPreview,
    PipelineConfig,
    QueryRequest,
    ReplayRequest,
)
from .vectorstore.store import VectorStore


app = FastAPI(title="Archivist RAG Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
ingestion_pipeline = IngestionPipeline(store)
metrics = MetricsRecorder(store)
pipelines = PipelineRegistry()

ui_path = Path(__file__).parent / "ui"
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")


def get_namespace(namespace: str | None) -> str:
    settings = get_settings()
    return namespace or settings.default_namespace


def parse_optional_int(value: str | None, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be an integer") from exc


@app.get("/", response_class=HTMLResponse)
async def root():
    return (ui_path / "index.html").read_text()


@app.post("/ingest/file")
async def ingest_file(
    collection: str = Form(...),
    namespace: str | None = Form(None),
    metadata: str | None = Form(None),
    chunk_size: str | None = Form(None),
    chunk_overlap: str | None = Form(None),
    file: UploadFile = File(...),
):
    meta_dict = {} if not metadata else {"user": metadata}
    parsed_chunk_size = parse_optional_int(chunk_size, "chunk_size")
    parsed_chunk_overlap = parse_optional_int(chunk_overlap, "chunk_overlap")
    content = await file.read()
    doc = ingestion_pipeline.ingest_file(
        file_name=file.filename,
        content=content,
        content_type=file.content_type or "text/plain",
        namespace=get_namespace(namespace),
        collection=collection,
        metadata=meta_dict,
        chunk_size=parsed_chunk_size,
        chunk_overlap=parsed_chunk_overlap,
    )
    return {"document_id": doc.id, "version": doc.version}


@app.post("/ingest/vhs")
async def ingest_vhs(
    video_url: str = Form(...),
    collection: str = Form(...),
    namespace: str | None = Form(None),
    metadata: str | None = Form(None),
):
    meta_dict = {} if not metadata else {"user": metadata}
    try:
        doc = ingestion_pipeline.ingest_video_link(
            video_url=video_url,
            namespace=get_namespace(namespace),
            collection=collection,
            metadata={"source": "vhs", **meta_dict},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return doc


def _execute_pipeline(request: QueryRequest) -> Dict[str, object]:
    namespace = get_namespace(request.namespace)
    start = time.time()
    chunks = store.search(
        namespace=namespace,
        collection=request.collection or "default",
        query=request.query,
        filters=request.filters,
    )
    pipeline = pipelines.get(request.pipeline)
    answer = "" if request.retrieval_only else pipeline.run(
        request.query, chunks, structured=request.structured, partial=request.partial
    )
    metrics.record(request.query, namespace, request.collection or "default", tokens=len(request.query.split()), start_time=start)
    return {"answer": answer, "chunks": chunks}


async def _streaming_response(result: Dict[str, object]):
    answer: str = result["answer"]
    for token in answer.split():
        yield token + " "
        await asyncio.sleep(0.01)


@app.post("/rag/query")
async def rag_query(request: QueryRequest):
    result = _execute_pipeline(request)
    if request.stream:
        return StreamingResponse(_streaming_response(result), media_type="text/plain")
    return result


@app.post("/rag/chat")
async def rag_chat(request: ChatRequest):
    combined = " ".join([m.content for m in request.history]) + " " + request.query
    chat_request = QueryRequest(**request.dict(), query=combined)
    result = _execute_pipeline(chat_request)
    if request.stream:
        return StreamingResponse(_streaming_response(result), media_type="text/plain")
    return result


@app.get("/pipelines")
async def list_pipelines() -> List[PipelineConfig]:
    return [PipelineConfig(name=name, description=p.description, optimized=p.optimized) for name, p in pipelines.pipelines.items()]


@app.post("/pipelines/{name}/optimize")
async def optimize_pipeline(name: str, feedback: Dict[str, str]):
    pipelines.optimize(name, feedback)
    return {"status": "optimized"}


@app.get("/collections", response_model=List[CollectionSummary])
async def list_collections(namespace: str | None = None):
    namespace_filter = namespace or None
    return [CollectionSummary(**summary) for summary in store.collections(namespace_filter)]


@app.get("/collections/{namespace}/{collection}", response_model=List[DocumentPreview])
async def get_collection_documents(namespace: str, collection: str):
    return [DocumentPreview(**doc) for doc in store.collection_documents(namespace, collection)]


@app.get("/admin/stats")
async def admin_stats():
    snapshot = metrics.snapshot()
    snapshot["latest_queries"] = store.latest_queries()
    return snapshot


@app.post("/admin/replay")
async def admin_replay(request: ReplayRequest):
    namespace = get_namespace(request.namespace)
    chunks = store.search(
        namespace=namespace,
        collection=request.collection or "default",
        query=request.query,
    )
    return {
        "query": request.query,
        "chunks": [
            {
                "id": c.id,
                "score": c.score,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
