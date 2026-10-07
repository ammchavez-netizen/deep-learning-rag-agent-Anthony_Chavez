"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import re
from dataclasses import fields, replace
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    # chunk_size / chunk_overlap are measured in *approximate tokens*
    # (~4 characters each). A deterministic estimate is used instead of a
    # real tokenizer so chunk boundaries — and therefore content-hash chunk
    # IDs — are identical on every machine, with no model download needed.
    CHARS_PER_TOKEN = 4

    SUPPORTED_MARKDOWN_SUFFIXES = (".md", ".markdown")
    SUPPORTED_PDF_SUFFIXES = (".pdf",)

    # Metadata vocabulary (mirrors the ChunkMetadata docstring in state.py).
    KNOWN_TOPICS = (
        "ANN", "CNN", "RNN", "LSTM", "Seq2Seq",
        "Autoencoder", "SOM", "BoltzmannMachine", "GAN",
    )
    BONUS_TOPICS = frozenset({"SOM", "BoltzmannMachine", "GAN"})
    VALID_DIFFICULTIES = ("beginner", "intermediate", "advanced")
    DEFAULT_TOPIC = "General"
    DEFAULT_DIFFICULTY = "intermediate"
    DEFAULT_TYPE = "concept_explanation"

    # Markdown headers that start a new section (and so a new chunk).
    MARKDOWN_HEADERS = (("#", "h1"), ("##", "h2"), ("###", "h3"))

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix in self.SUPPORTED_PDF_SUFFIXES:
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix in self.SUPPORTED_MARKDOWN_SUFFIXES:
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            supported = self.SUPPORTED_PDF_SUFFIXES + self.SUPPORTED_MARKDOWN_SUFFIXES
            raise ValueError(
                f"Unsupported file type '{suffix}' for {file_path.name}. "
                f"Supported: {', '.join(supported)}"
            )

        base_metadata = self._infer_metadata(file_path, metadata_overrides)

        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            text = raw["text"].strip()
            if not text:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=VectorStoreManager.generate_chunk_id(
                        base_metadata.source, text
                    ),
                    chunk_text=text,
                    # Each chunk owns its metadata object (and its list).
                    metadata=replace(
                        base_metadata,
                        related_topics=list(base_metadata.related_topics),
                    ),
                )
            )

        logger.info(
            "Chunked '{}' -> {} chunk(s) | topic={} difficulty={}",
            file_path.name,
            len(chunks),
            base_metadata.topic,
            base_metadata.difficulty,
        )
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        all_chunks: list[DocumentChunk] = []
        for file_path in file_paths:
            try:
                all_chunks.extend(
                    self.chunk_file(Path(file_path), metadata_overrides)
                )
            except Exception as exc:
                # One unreadable file must not abort the rest of the upload.
                logger.error("Skipping '{}': {}", file_path, exc)
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        pages = PyPDFLoader(str(file_path)).load()
        total_pages = len(pages)

        cleaned_pages = []
        for page in pages:
            page_index = page.metadata.get("page", 0)
            text, hit_references = self._clean_pdf_text(
                page.page_content,
                allow_reference_cut=page_index >= total_pages / 2,
            )
            if text:
                page.page_content = text
                cleaned_pages.append(page)
            if hit_references:
                break  # everything after the reference list is noise

        splitter = self._build_splitter(chunk_size, chunk_overlap)
        return [
            {"text": doc.page_content, "page": doc.metadata.get("page", 0) + 1}
            for doc in splitter.split_documents(cleaned_pages)
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        markdown = file_path.read_text(encoding="utf-8")

        # Pass 1: split on headers so each section stays in one piece.
        # strip_headers=False keeps the topic name inside the chunk text,
        # which helps the embedding capture what the chunk is about.
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(self.MARKDOWN_HEADERS),
            strip_headers=False,
        )
        sections = header_splitter.split_text(markdown)

        # Pass 2: only sections larger than chunk_size get split further.
        splitter = self._build_splitter(chunk_size, chunk_overlap)

        raw_chunks = []
        for section in sections:
            if not self._has_body_text(section.page_content):
                continue  # a header with nothing under it is not a chunk
            header = " > ".join(
                section.metadata[key]
                for _, key in self.MARKDOWN_HEADERS
                if key in section.metadata
            )
            for piece in self._split_section(section.page_content, splitter, chunk_size):
                raw_chunks.append({"text": piece, "header": header})
        return raw_chunks

    def _split_section(
        self,
        section: str,
        splitter: RecursiveCharacterTextSplitter,
        chunk_size: int,
    ) -> list[str]:
        """
        Split one markdown section only if it exceeds chunk_size.

        When it must be split, the header lines are split off first and
        re-attached to every piece. Otherwise the header would be left as
        a tiny orphan chunk and the later pieces would lose their topic.
        """
        if len(section) // self.CHARS_PER_TOKEN <= chunk_size:
            return [section]

        lines = section.splitlines()
        body_start = next(
            i for i, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("#")
        )
        header_block = "\n".join(lines[:body_start]).strip()
        body = "\n".join(lines[body_start:])
        return [f"{header_block}\n{piece}" for piece in splitter.split_text(body)]

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        overrides = dict(overrides or {})

        valid_keys = {f.name for f in fields(ChunkMetadata)}
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown metadata override(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_keys)}"
            )

        # --- Infer from the filename: <topic>_<difficulty>.<ext> ---
        stem = file_path.stem
        topic_part, _, last_part = stem.rpartition("_")
        if topic_part and last_part.lower() in self.VALID_DIFFICULTIES:
            difficulty = last_part.lower()
        else:
            topic_part, difficulty = stem, self.DEFAULT_DIFFICULTY

        metadata = {
            "topic": self._match_topic(topic_part) or self.DEFAULT_TOPIC,
            "difficulty": difficulty,
            "type": self.DEFAULT_TYPE,
            "source": file_path.name,
            "related_topics": [],
        }

        # --- Explicit overrides win over inference ---
        metadata.update(overrides)

        if isinstance(metadata["topic"], str):
            # Accept "lstm" and store it as the canonical "LSTM".
            metadata["topic"] = self._match_topic(metadata["topic"]) or metadata["topic"]

        metadata["difficulty"] = str(metadata["difficulty"]).lower()
        if metadata["difficulty"] not in self.VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{metadata['difficulty']}'. "
                f"Expected one of {self.VALID_DIFFICULTIES}"
            )

        related = metadata["related_topics"]
        if isinstance(related, str):
            related = [t.strip() for t in related.split(",") if t.strip()]
        metadata["related_topics"] = list(related)

        # Bonus status follows the topic unless explicitly overridden.
        if "is_bonus" not in overrides:
            metadata["is_bonus"] = metadata["topic"] in self.BONUS_TOPICS

        return ChunkMetadata(**metadata)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_splitter(
        self, chunk_size: int, chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        """Build a recursive splitter that measures size in approximate tokens."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(text) // self.CHARS_PER_TOKEN,
        )

    @staticmethod
    def _normalise_key(text: str) -> str:
        """Lower-case and strip punctuation so 'Seq_2_Seq' matches 'Seq2Seq'."""
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _match_topic(self, text: str) -> str | None:
        """Return the canonical topic named by `text` (or by one of its
        underscore-separated words), or None if no known topic matches."""
        lookup = {self._normalise_key(t): t for t in self.KNOWN_TOPICS}
        lookup["boltzmann"] = "BoltzmannMachine"

        for candidate in [text, *text.split("_")]:
            topic = lookup.get(self._normalise_key(candidate))
            if topic:
                return topic
        return None

    @staticmethod
    def _has_body_text(section: str) -> bool:
        """True if a markdown section has content beyond its header lines."""
        return any(
            line.strip() and not line.lstrip().startswith("#")
            for line in section.splitlines()
        )

    @staticmethod
    def _clean_pdf_text(
        text: str, allow_reference_cut: bool
    ) -> tuple[str, bool]:
        """
        Strip common academic-PDF noise from one page of extracted text.

        Removes bare page-number lines and collapses runs of spaces. If
        `allow_reference_cut` is set and a standalone 'References' heading
        is found, everything from it onward is dropped.

        Returns
        -------
        tuple[str, bool]
            (cleaned text, whether a reference list was cut off)
        """
        kept_lines = []
        hit_references = False
        for line in text.splitlines():
            stripped = line.strip()
            if allow_reference_cut and re.fullmatch(
                r"(references|bibliography)", stripped, flags=re.IGNORECASE
            ):
                hit_references = True
                break
            if re.fullmatch(r"\d{1,4}", stripped):
                continue  # page number
            kept_lines.append(re.sub(r"[ \t]+", " ", stripped))
        return "\n".join(kept_lines).strip(), hit_references
