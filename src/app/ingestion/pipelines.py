import hashlib
import time
from typing import Dict, Iterable, List, Tuple

from ..config import get_settings
from ..models import Chunk, Document
from ..vectorstore.store import VectorStore
from .cleaning import chunk_text, clean_text, detect_language
from .loaders import extract_zip, fetch_api, load_file, load_git_repo, parse_webhook, scrape_url


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

    def ingest_url(
        self, *,
        url: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        text, meta = scrape_url(url)
        return self._chunk_and_index(
            content=text,
            namespace=namespace,
            collection=collection,
            metadata={**metadata, **meta},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_git(
        self, *,
        repo_url: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
        branch: str | None = None,
        path: str | None = None,
    ) -> List[Document]:
        documents = []
        for file_name, text in load_git_repo(repo_url, branch=branch, path=path):
            doc = self._chunk_and_index(
                content=text,
                namespace=namespace,
                collection=collection,
                metadata={**metadata, "source_file": file_name, "source_repo": repo_url},
                chunk_size=None,
                chunk_overlap=None,
            )
            documents.append(doc)
        return documents

    def ingest_api(
        self, *,
        endpoint: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
        headers: Dict[str, str],
    ) -> Document:
        text = fetch_api(endpoint, headers)
        return self._chunk_and_index(
            content=text,
            namespace=namespace,
            collection=collection,
            metadata={**metadata, "source_api": endpoint},
            chunk_size=None,
            chunk_overlap=None,
        )

    def ingest_webhook(
        self, *,
        payload: Dict[str, str],
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
    ) -> Document:
        text = parse_webhook(payload)
        return self._chunk_and_index(
            content=text,
            namespace=namespace,
            collection=collection,
            metadata=metadata,
            chunk_size=None,
            chunk_overlap=None,
        )

    def ingest_video_transcript(
        self,
        *,
        transcript: str,
        namespace: str,
        collection: str,
        metadata: Dict[str, str],
    ) -> Document:
        return self._chunk_and_index(
            content=transcript,
            namespace=namespace,
            collection=collection,
            metadata={**metadata, "source": "vhs"},
            chunk_size=None,
            chunk_overlap=None,
        )
