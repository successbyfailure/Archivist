import time
from typing import Dict

from .vectorstore.store import VectorStore


class MetricsRecorder:
    def __init__(self, store: VectorStore):
        self.store = store
        self.tokens_used = 0
        self.latencies: list[float] = []

    def record(self, query: str, namespace: str, collection: str, tokens: int, start_time: float):
        latency_ms = (time.time() - start_time) * 1000
        self.latencies.append(latency_ms)
        self.tokens_used += tokens
        self.store.log_query(query, latency_ms, tokens, namespace, collection)

    def snapshot(self) -> Dict[str, float]:
        stats = self.store.stats()
        latency = sum(self.latencies[-100:]) / len(self.latencies[-100:]) if self.latencies else 0.0
        return {
            **stats,
            "latency_ms": latency,
            "tokens_used": self.tokens_used,
        }
