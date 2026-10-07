"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    # Chunks embedded and written per call — bounds memory use on large corpora.
    INGEST_BATCH_SIZE = 100

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        db_path = Path(self._settings.chroma_db_path)
        try:
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialise ChromaDB at '{db_path}': {exc}"
            ) from exc

        logger.info(
            "ChromaDB ready | collection='{}' | path='{}' | items={}",
            self._settings.chroma_collection_name,
            db_path,
            self._collection.count(),
        )

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()

        # 1. Duplicate detection. Also catches repeats *within* this call:
        #    ChromaDB rejects an upsert that contains the same ID twice.
        new_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen_ids or self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue
            seen_ids.add(chunk.chunk_id)
            new_chunks.append(chunk)

        # 2. Embed and store in batches.
        ingested_sources: set[str] = set()
        for start in range(0, len(new_chunks), self.INGEST_BATCH_SIZE):
            batch = new_chunks[start : start + self.INGEST_BATCH_SIZE]
            try:
                self._upsert_batch(batch)
                stored = batch
            except Exception as exc:
                # One bad chunk should not sink its whole batch — retry
                # individually so each failure gets its own error message.
                logger.warning("Batch upsert failed ({}); retrying per chunk", exc)
                stored = []
                for chunk in batch:
                    try:
                        self._upsert_batch([chunk])
                        stored.append(chunk)
                    except Exception as chunk_exc:
                        result.errors.append(
                            f"{chunk.metadata.source} [{chunk.chunk_id}]: {chunk_exc}"
                        )

            result.ingested += len(stored)
            ingested_sources.update(c.metadata.source for c in stored)

        result.document_ids = sorted(ingested_sources)

        logger.info(
            "Ingestion complete | ingested={} skipped={} errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    def _upsert_batch(self, batch: list[DocumentChunk]) -> None:
        """Embed a batch of chunks in one call and upsert them into ChromaDB."""
        embeddings = self._embeddings.embed_documents([c.chunk_text for c in batch])
        self._collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.chunk_text for c in batch],
            metadatas=[c.metadata.to_dict() for c in batch],
        )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        # A blank query has nothing to embed — return "no context" gracefully.
        if not query_text or not query_text.strip():
            return []

        k = k or self._settings.retrieval_k

        total = self._collection.count()
        if total == 0:
            return []

        # ChromaDB wants a single condition as-is, several wrapped in $and.
        conditions = []
        if topic_filter:
            conditions.append({"topic": topic_filter})
        if difficulty_filter:
            conditions.append({"difficulty": difficulty_filter})
        if not conditions:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        query_embedding = self._embeddings.embed_query(query_text)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Results are nested one level per query embedding; we sent exactly one.
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance.
            score = max(0.0, min(1.0, 1.0 - distance))
            if score < self._settings.similarity_threshold:
                continue  # hallucination guard: drop weak matches
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(meta),
                    score=score,
                )
            )

        retrieved.sort(key=lambda c: c.score, reverse=True)
        logger.debug(
            "Query '{}' -> {} chunk(s) above threshold {}",
            query_text[:60],
            len(retrieved),
            self._settings.similarity_threshold,
        )
        return retrieved

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]

        documents: dict[str, dict] = {}
        for meta in metadatas:
            source = meta["source"]
            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": meta["topic"],
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1

        return sorted(documents.values(), key=lambda d: d["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        raw = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        # ChromaDB returns rows in insertion order, and ingest() writes chunks
        # in document order, so this preserves position in the source file.
        return [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=text,
                metadata=ChunkMetadata.from_dict(meta),
            )
            for chunk_id, text, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        ]

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]

        return {
            "total_chunks": len(metadatas),
            "topics": sorted({m["topic"] for m in metadatas}),
            "sources": sorted({m["source"] for m in metadatas}),
            "bonus_topics_present": any(
                str(m.get("is_bonus", "false")).lower() == "true" for m in metadatas
            ),
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        ids = self._collection.get(where={"source": source})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Deleted {} chunk(s) from '{}'", len(ids), source)
        return len(ids)
