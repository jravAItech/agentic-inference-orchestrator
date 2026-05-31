"""
Azure Blob Storage backend for runbook corpus + incident inputs.

Mirrors a real production pattern (documents land in Blob, services read from
it) instead of bundling files in the repo. Opt-in: the local filesystem stays
the default, so the project still runs $0 with no Azure account.

Enable with:
    export CORPUS_BACKEND=azure_blob
    export AZURE_STORAGE_CONNECTION_STRING=...
    export AZURE_BLOB_CONTAINER=runbooks
"""

from __future__ import annotations
import os


class BlobCorpus:
    """Loads runbook documents from an Azure Blob container."""

    def __init__(self, container: str | None = None,
                 conn_str: str | None = None):
        self.container = container or os.environ.get(
            "AZURE_BLOB_CONTAINER", "runbooks")
        self.conn_str = conn_str or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING", "")

    def load(self) -> list[str]:
        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient.from_connection_string(self.conn_str)
        container = svc.get_container_client(self.container)
        docs = []
        for blob in container.list_blobs():
            if blob.name.endswith(".md"):
                data = container.download_blob(blob.name).readall()
                docs.append(data.decode("utf-8"))
        return docs

    def upload_seed_corpus(self, local_dir: str = "rag/corpus") -> int:
        """One-time helper: push the local sample runbooks into Blob."""
        import glob
        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient.from_connection_string(self.conn_str)
        try:
            svc.create_container(self.container)
        except Exception:
            pass  # already exists
        container = svc.get_container_client(self.container)
        count = 0
        for path in glob.glob(os.path.join(local_dir, "*.md")):
            name = os.path.basename(path)
            with open(path, "rb") as f:
                container.upload_blob(name, f, overwrite=True)
            count += 1
        return count
