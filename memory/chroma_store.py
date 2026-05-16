"""
ChromaDB semantic memory store.

Stores full reasoning traces from past agent runs as text embeddings.
At the start of each new run, retrieves the top-K most semantically
similar past traces to inject as context into the system prompt.

Why ChromaDB and not SQLite?
    SQLite can store text but cannot answer "find me traces about topics
    similar to this new query" — that requires vector similarity search.
    ChromaDB stores embeddings and answers nearest-neighbour queries in
    milliseconds, even with thousands of stored traces.
"""

import logging
from typing import Optional

import chromadb
import yaml
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ChromaStore:

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        mem_cfg = cfg["memory"]["chroma"]

        self._top_k = cfg["agent"]["memory_recall_top_k"]

        # Persistent local ChromaDB client — data survives restarts
        self._client = chromadb.PersistentClient(path=mem_cfg["persist_directory"])

        # sentence-transformers embedding function (runs locally, no API key)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=mem_cfg["embedding_model"]
        )

        self._collection = self._client.get_or_create_collection(
            name=mem_cfg["collection_name"],
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "chroma_initialized",
            extra={
                "collection": mem_cfg["collection_name"],
                "persist_dir": mem_cfg["persist_directory"],
                "existing_traces": self._collection.count(),
            },
        )

    def store_trace(self, run_id: str, query: str, trace_summary: str) -> None:
        """
        Store a completed agent run's reasoning trace.

        Args:
            run_id:        UUID of the run — used as the ChromaDB document ID.
            query:         The original user question (stored as metadata).
            trace_summary: A plain-text summary of the reasoning steps taken.
                           This is what gets embedded and searched against.
        """
        try:
            self._collection.upsert(
                ids=[run_id],
                documents=[trace_summary],
                metadatas=[{"query": query, "run_id": run_id}],
            )
            logger.debug(
                "trace_stored",
                extra={"run_id": run_id, "chars": len(trace_summary)},
            )
        except Exception as e:
            # Memory storage failure must never crash the agent
            logger.error("chroma_store_failed", extra={"run_id": run_id, "error": str(e)})

    def retrieve_similar(self, query: str) -> list[str]:
        """
        Retrieve the top-K past traces most semantically similar to query.

        Returns an empty list if the collection is empty or an error occurs.
        The agent loop treats an empty list as "no relevant memory" and continues.
        """
        count = self._collection.count()
        if count == 0:
            logger.debug("chroma_empty_collection")
            return []

        # Can't request more results than documents in the collection
        k = min(self._top_k, count)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            formatted = []
            for doc, meta, dist in zip(docs, metas, distances):
                similarity = round(1 - dist, 3)   # cosine distance → similarity
                header = (
                    f"[Past query: {meta.get('query', 'unknown')} | "
                    f"similarity: {similarity}]"
                )
                formatted.append(f"{header}\n{doc}")

            logger.debug(
                "chroma_retrieved",
                extra={"query": query[:80], "k": k, "returned": len(formatted)},
            )
            return formatted

        except Exception as e:
            logger.error("chroma_retrieve_failed", extra={"error": str(e)})
            return []

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def build_trace_summary(query: str, steps: list[dict], outcome: str) -> str:
        """
        Build a plain-text trace summary suitable for embedding.

        Stored in ChromaDB; retrieved as memory context for future runs.
        Keeps it concise — just enough for the LLM to learn from it.
        """
        lines = [f"Query: {query}", f"Outcome: {outcome}", "Steps:"]
        for i, step in enumerate(steps, 1):
            lines.append(
                f"  {i}. Thought: {step.get('thought', '')[:100]}"
            )
            lines.append(
                f"     Action: {step.get('action', '')} — "
                f"Input: {step.get('action_input', '')[:80]}"
            )
        return "\n".join(lines)