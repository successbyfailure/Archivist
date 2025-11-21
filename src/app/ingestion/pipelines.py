import hashlib
import time
from typing import Dict

import requests

from ..config import get_settings
from ..models import Chunk, Document
from ..vectorstore.store import VectorStore

from .cleaning import chunk_text, clean_text, detect_language
from .loaders import extract_zip, load_file


def _normalize_vhs_text_response(response: requests.Response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        if isinstance(payload, dict) and "segments" in payload:
            return " ".join(seg.get("text", "") for seg in payload.get("segments", [])).strip()
        if isinstance(payload, dict) and "text" in payload:
            return str(payload["text"]).strip()
        error_msg = payload.get("error") if isinstance(payload, dict) else None
        raise ValueError(error_msg or f"Unexpected VHS response: {payload}")

    text_content = response.text.strip()
    if not text_content:
        raise ValueError("Empty transcript received from VHS")
    return text_content


def fetch_vhs_transcript(video_url: str) -> str:
    settings = get_settings()
    if not settings.enable_vhs:
        raise ValueError("VHS integration is disabled")
    if not settings.vhs_endpoint:
        raise ValueError("VHS endpoint is not configured")

    headers = {}
    if settings.vhs_api_key:
        headers["Authorization"] = f"Bearer {settings.vhs_api_key}"

    try:
        response = requests.get(
            f"{settings.vhs_endpoint.rstrip('/')}/api/download",
            params={"url": video_url, "format": "transcripcion_txt"},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Error requesting VHS transcript: {exc}") from exc

    transcript = _normalize_vhs_text_response(response)
    if not transcript:
        raise ValueError("VHS response did not include a transcript")
    return transcript


class IngestionPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    def _chunk_and_index(
        self, *,
        content: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
        chunk_size: int | None,
        chunk_overlap: int | None,
    ) -> Document:
        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap or settings.chunk_overlap
        language = detect_language(content)
        chunks = []
        for idx, chunk in enumerate(chunk_text(content, size=size, overlap=overlap)):
            text_clean = clean_text(chunk)
            chunk_id = hashlib.sha1(f"{collection}-{idx}-{time.time()}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text_clean,
                    embedding=self.store.embed(text_clean),
                    metadata={"collection": collection, **metadata},
                )
            )
        document = self.store.add_document(
            collection=collection,
            namespace=namespace,
            content=content,
            chunks=chunks,
            metadata={**metadata, "language": language},
            language=language,
        )
        return document

    def ingest_file(
        self, *,
        file_name: str,
        content: bytes,
        content_type: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        if file_name.endswith(".zip"):
            documents = []
            for inner_name, inner_content in extract_zip(content):
                doc = self.ingest_file(
                    file_name=inner_name,
                    content=inner_content,
                    content_type=content_type,
                    namespace=namespace,
                    collection=collection,
                    metadata={"archive": file_name, **metadata},
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                documents.append(doc)
            return documents[-1]

        text = load_file(file_name, content, content_type)
        return self._chunk_and_index(
            content=text,
            namespace=namespace,
            collection=collection,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_video_link(
        self,
        *,
        video_url: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
    ) -> Document:
        transcript = fetch_vhs_transcript(video_url)
        return self._chunk_and_index(
            content=transcript,
            namespace=namespace,
            collection=collection,
            metadata={**metadata, "source": "vhs", "video_url": video_url},
            chunk_size=None,
            chunk_overlap=None,
        )
