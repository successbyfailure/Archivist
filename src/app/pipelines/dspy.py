from typing import Dict, List

from ..models import Chunk


class DSPyPipeline:
    def __init__(self, name: str, description: str, optimized: bool = False):
        self.name = name
        self.description = description
        self.optimized = optimized

    def run(self, query: str, chunks: List[Chunk], structured: bool = False, partial: bool = False) -> str:
        citations = [f"[{c.metadata.get('source_file') or c.metadata.get('source_api') or c.metadata.get('collection','')}:{c.id[:6]}]" for c in chunks]
        base_answer = f"Answer for '{query}' using pipeline '{self.name}'."
        if structured:
            return base_answer + " Structured response." + " Citations: " + " ".join(citations)
        if partial:
            return base_answer + " Partial response." + " Citations: " + " ".join(citations)
        return base_answer + " Citations: " + " ".join(citations)

    def optimize(self, feedback: Dict[str, str]):
        self.optimized = True


class PipelineRegistry:
    def __init__(self):
        self.pipelines: Dict[str, DSPyPipeline] = {
            "qa": DSPyPipeline("qa", "QA with citations"),
            "reasoning": DSPyPipeline("reasoning", "Multi-step reasoning"),
            "routing": DSPyPipeline("routing", "Query routing"),
            "extraction": DSPyPipeline("extraction", "Structured extraction/classification"),
        }

    def get(self, name: str | None) -> DSPyPipeline:
        if name and name in self.pipelines:
            return self.pipelines[name]
        return self.pipelines["qa"]

    def optimize(self, name: str, feedback: Dict[str, str]):
        pipeline = self.get(name)
        pipeline.optimize(feedback)
