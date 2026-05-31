"""
Minimal hybrid retriever over a local runbook corpus. Embedding-based
semantic search with a keyword fallback — the same hybrid lesson that
mattered in production (engineers search by codes/IDs, not just intent).

Phase-1 friendly: uses sentence-transformers (CPU-fine). Swap in a vector
DB (OpenSearch / Azure AI Search) for scale without touching the agents.
"""

from __future__ import annotations
import os
import glob


class RunbookRetriever:
    def __init__(self, corpus_dir: str = "rag/corpus"):
        self.corpus_dir = corpus_dir
        self.docs = self._load()
        self._model = None
        self._embeddings = None

    def _load(self) -> list[str]:
        paths = glob.glob(os.path.join(self.corpus_dir, "*.md"))
        docs = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                docs.append(f.read())
        return docs

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = self._model.encode(self.docs)

    def search(self, query: str, k: int = 4) -> list[str]:
        if not self.docs:
            return []
        # Keyword pass (catches part numbers / fault codes / IDs).
        kw = [d for d in self.docs
              if any(tok.lower() in d.lower() for tok in query.split())]
        # Semantic pass.
        self._ensure_model()
        import numpy as np
        q = self._model.encode([query])[0]
        sims = self._embeddings @ q / (
            (self._embeddings**2).sum(1)**0.5 * (q**2).sum()**0.5 + 1e-9)
        ranked = [self.docs[i] for i in np.argsort(sims)[::-1]]
        # Merge, keyword hits first, dedup, top-k.
        merged, seen = [], set()
        for d in kw + ranked:
            if d not in seen:
                merged.append(d)
                seen.add(d)
        return merged[:k]


def get_retriever(corpus_dir: str = "rag/corpus"):
    """
    Factory. Returns the retriever backend selected by env var:
      RETRIEVER_BACKEND=local         -> local sentence-transformers (default)
      RETRIEVER_BACKEND=azure_search  -> managed Azure AI Search (hybrid)

    The local backend can still source its corpus from Azure Blob when
    CORPUS_BACKEND=azure_blob (see cloud/azure_blob.py).
    """
    import os
    backend = os.environ.get("RETRIEVER_BACKEND", "local")
    if backend == "azure_search":
        from cloud.azure_search import AzureSearchRetriever
        return AzureSearchRetriever()

    r = RunbookRetriever(corpus_dir)
    if os.environ.get("CORPUS_BACKEND") == "azure_blob":
        from cloud.azure_blob import BlobCorpus
        r.docs = BlobCorpus().load()
    return r
