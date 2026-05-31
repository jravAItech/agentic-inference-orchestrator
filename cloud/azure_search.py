"""
Azure AI Search backend — managed hybrid (semantic + keyword) retrieval.

This is the same hybrid pattern that matters in production: operators search
by fault codes / IDs (keyword) as well as intent (semantic/vector). Azure AI
Search runs both in one query and fuses the results.

Opt-in. The local sentence-transformers retriever stays the default.

Enable with:
    export RETRIEVER_BACKEND=azure_search
    export AZURE_SEARCH_ENDPOINT=https://<svc>.search.windows.net
    export AZURE_SEARCH_KEY=...
    export AZURE_SEARCH_INDEX=runbooks
"""

from __future__ import annotations
import os


class AzureSearchRetriever:
    """Hybrid retrieval over an Azure AI Search index."""

    def __init__(self, endpoint: str | None = None, key: str | None = None,
                 index: str | None = None):
        self.endpoint = endpoint or os.environ["AZURE_SEARCH_ENDPOINT"]
        self.key = key or os.environ["AZURE_SEARCH_KEY"]
        self.index = index or os.environ.get("AZURE_SEARCH_INDEX", "runbooks")
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential
            self._client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index,
                credential=AzureKeyCredential(self.key),
            )
        return self._client

    def search(self, query: str, k: int = 4) -> list[str]:
        client = self._client_lazy()
        # query_type='semantic' + a search term gives hybrid keyword+semantic.
        results = client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            top=k,
        )
        return [doc.get("content", "") for doc in results]
