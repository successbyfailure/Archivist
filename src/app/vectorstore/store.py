import math
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np

from ..models import Chunk, Document


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.split() if tok.strip()]


class VectorStore:
    def __init__(self):
        self.documents: Dict[str, List[Document]] = defaultdict(list)
        self.vectors: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.bm25_index: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        self.doc_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.metrics_window = deque(maxlen=1000)
        self.relationships: Dict[str, List[str]] = defaultdict(list)

    def _namespace_key(self, namespace: str, collection: str) -> str:
        return f"{namespace}:{collection}"

    def embed(self, text: str) -> List[float]:
        # A lightweight embedding using hashing for determinism
        tokens = _tokenize(text)
        vec = np.zeros(128)
        for token in tokens:
            idx = hash(token) % 128
            vec[idx] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _index_bm25(self, namespace: str, collection: str, chunk: Chunk):
        key = self._namespace_key(namespace, collection)
        tokens = _tokenize(chunk.text)
        for token in set(tokens):
            self.bm25_index[key][chunk.id][token] = tokens.count(token)
            self.doc_freq[key][token] += 1

    def add_document(
        self,
        *,
        namespace: str,
        collection: str,
        content: str,
        chunks: List[Chunk],
        metadata: Dict[str, str],
        language: str | None = None,
    ) -> Document:
        key = self._namespace_key(namespace, collection)
        version = len(self.documents[key]) + 1
        doc = Document(
            id=str(uuid.uuid4()),
            namespace=namespace,
            collection=collection,
            version=version,
            content=content,
            chunks=chunks,
            metadata=metadata,
            language=language,
        )
        self.documents[key].append(doc)
        for chunk in chunks:
            self.vectors[key][chunk.id] = np.array(chunk.embedding)
            self._index_bm25(namespace, collection, chunk)
        return doc

    def _bm25_score(self, query_tokens: List[str], namespace: str, collection: str) -> Dict[str, float]:
        key = self._namespace_key(namespace, collection)
        scores: Dict[str, float] = defaultdict(float)
        doc_count = len(self.documents.get(key, [])) or 1
        avgdl = np.mean([len(_tokenize(c.text)) for doc in self.documents.get(key, []) for c in doc.chunks] or [1])
        for doc in self.documents.get(key, []):
            for chunk in doc.chunks:
                freq_map = self.bm25_index[key].get(chunk.id, {})
                dl = len(_tokenize(chunk.text)) or 1
                for token in query_tokens:
                    tf = freq_map.get(token, 0)
                    idf = math.log((doc_count - self.doc_freq[key].get(token, 0) + 0.5) / (self.doc_freq[key].get(token, 0) + 0.5) + 1)
                    score = idf * ((tf * 2.0) / (tf + 1.5 * (0.25 + 0.75 * (dl / avgdl))))
                    scores[chunk.id] += score
        return scores

    def search(
        self,
        *,
        namespace: str,
        collection: str,
        query: str,
        top_k: int = 5,
        filters: Dict[str, str] | None = None,
    ) -> List[Chunk]:
        key = self._namespace_key(namespace, collection)
        filters = filters or {}
        query_vec = np.array(self.embed(query))
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25_score(query_tokens, namespace, collection)

        scored: List[Tuple[str, float]] = []
        for doc in self.documents.get(key, []):
            for chunk in doc.chunks:
                if any(chunk.metadata.get(k) != v for k, v in filters.items()):
                    continue
                vec = self.vectors[key].get(chunk.id)
                if vec is None:
                    continue
                sim = float(np.dot(query_vec, vec))
                hybrid = 0.6 * sim + 0.4 * bm25_scores.get(chunk.id, 0.0)
                scored.append((chunk.id, hybrid))
                chunk.score = hybrid
        scored.sort(key=lambda x: x[1], reverse=True)
        chunk_lookup = {c.id: c for doc in self.documents.get(key, []) for c in doc.chunks}
        return [chunk_lookup[cid] for cid, _ in scored[:top_k]]

    def track_relationship(self, source_id: str, related_id: str):
        self.relationships[source_id].append(related_id)

    def stats(self) -> Dict[str, float]:
        total_chunks = sum(len(doc.chunks) for docs in self.documents.values() for doc in docs)
        vector_size = sum(len(vectors) for vectors in self.vectors.values())
        avg_length = np.mean([
            len(_tokenize(chunk.text)) for docs in self.documents.values() for doc in docs for chunk in doc.chunks
        ] or [0])
        return {
            "documents": sum(len(docs) for docs in self.documents.values()),
            "chunks": total_chunks,
            "vector_size": vector_size,
            "avg_chunk_length": float(avg_length),
        }

    def collections(self, namespace: str | None = None) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for key, docs in self.documents.items():
            ns, collection = key.split(":", 1)
            if namespace and ns != namespace:
                continue
            chunk_count = sum(len(doc.chunks) for doc in docs)
            languages = sorted(
                {doc.language or doc.metadata.get("language") for doc in docs if doc.language or doc.metadata.get("language")}
            )
            latest = max(docs, key=lambda d: d.created_at) if docs else None
            summaries.append(
                {
                    "namespace": ns,
                    "collection": collection,
                    "documents": len(docs),
                    "chunks": chunk_count,
                    "latest_version": latest.version if latest else 0,
                    "languages": languages,
                    "latest_ingested_at": latest.created_at if latest else None,
                }
            )
        summaries.sort(key=lambda s: (s["namespace"], s["collection"]))
        return summaries

    def collection_documents(self, namespace: str, collection: str) -> List[Dict[str, object]]:
        key = self._namespace_key(namespace, collection)
        previews: List[Dict[str, object]] = []
        for doc in self.documents.get(key, []):
            snippet = " ".join(chunk.text for chunk in doc.chunks[:1])
            previews.append(
                {
                    "id": doc.id,
                    "version": doc.version,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at,
                    "chunks": len(doc.chunks),
                    "snippet": snippet[:280],
                    "language": doc.language or doc.metadata.get("language"),
                }
            )
        previews.sort(key=lambda d: d["version"], reverse=True)
        return previews

    def log_query(self, query: str, latency_ms: float, tokens: int, namespace: str, collection: str):
        self.metrics_window.append(
            {
                "query": query,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "namespace": namespace,
                "collection": collection,
            }
        )

    def latest_queries(self) -> List[Dict[str, str]]:
        return list(self.metrics_window)
