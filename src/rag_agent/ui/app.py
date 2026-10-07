"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


ALL_FILTER = "All"
DIFFICULTY_OPTIONS = [ALL_FILTER, "beginner", "intermediate", "advanced"]


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    # Always show what is really in the store (it persists between restarts).
    st.session_state.ingested_documents = store.list_documents()

    # The uploader key changes after each ingestion, which clears the widget.
    uploader_key = f"uploader_{st.session_state.get('uploader_nonce', 0)}"
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    if st.sidebar.button(
        "Ingest Documents",
        type="primary",
        disabled=not uploaded_files,
    ):
        with st.spinner("Chunking and embedding documents..."):
            result = _ingest_uploads(uploaded_files, store, chunker)
        st.session_state.last_ingestion_result = result
        st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
        st.rerun()

    _render_ingestion_result(st.session_state.last_ingestion_result)

    # Ingested documents list, with a remove button for each.
    docs = st.session_state.ingested_documents
    if docs:
        st.sidebar.subheader("Ingested documents")
    for doc in docs:
        name_col, button_col = st.sidebar.columns([5, 1])
        name_col.markdown(
            f"**{doc['source']}**  \n{doc['topic']} · {doc['chunk_count']} chunks"
        )
        if button_col.button("🗑", key=f"remove_{doc['source']}", help="Remove"):
            store.delete_document(doc["source"])
            if st.session_state.selected_document == doc["source"]:
                st.session_state.selected_document = None
            st.session_state.last_ingestion_result = None
            st.rerun()

    if not docs:
        st.sidebar.info("Upload .pdf or .md files to populate the corpus.")


def _ingest_uploads(uploaded_files, store: VectorStoreManager, chunker: DocumentChunker):
    """
    Save uploads to a temp directory, chunk them, and ingest the chunks.

    The temp files keep their original names so each chunk's `source`
    metadata matches what the user uploaded. Files that produce no chunks
    (unreadable, unsupported, empty) are reported in result.errors.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for uploaded in uploaded_files:
            path = Path(tmp_dir) / Path(uploaded.name).name  # drop any directories
            path.write_bytes(uploaded.getbuffer())
            paths.append(path)

        chunks = chunker.chunk_files(paths)

    result = store.ingest(chunks)

    chunked_sources = {c.metadata.source for c in chunks}
    for uploaded in uploaded_files:
        if Path(uploaded.name).name not in chunked_sources:
            result.errors.append(f"{uploaded.name}: no readable content found")
    return result


def _render_ingestion_result(result) -> None:
    """Show the outcome of the last ingestion as success / warning / error."""
    if result is None:
        return

    summary = f"{result.ingested} chunks added, {result.skipped} duplicates skipped"
    if result.errors:
        st.sidebar.error(f"{summary}, {len(result.errors)} error(s)")
        with st.sidebar.expander("Error details"):
            for message in result.errors:
                st.caption(message)
    elif result.ingested > 0:
        st.sidebar.success(summary)
    else:
        st.sidebar.warning(f"Nothing new to add — {summary}")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.divider()
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write("Topics:", ", ".join(stats["topics"]) or "none yet")
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    docs = st.session_state.ingested_documents
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in docs]
    current = st.session_state.selected_document
    index = sources.index(current) if current in sources else 0

    # Separate widget key: the "selected_document" default is None, which
    # is not a valid selectbox value.
    selected = st.selectbox("Select document", options=sources, index=index,
                            key="document_selectbox")
    st.session_state.selected_document = selected

    chunks = store.get_document_chunks(selected)
    scores = st.session_state.get("last_retrieved_scores", {})

    viewer = st.container(height=400)
    with viewer:
        for number, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            st.markdown(
                f"**Chunk {number}** · `{meta.topic}` · `{meta.difficulty}` "
                f"· `{meta.type}`"
            )
            st.write(chunk.chunk_text)
            if chunk.chunk_id in scores:
                st.caption(f"🎯 Used in last response — similarity {scores[chunk.chunk_id]:.2f}")
            st.divider()

    word_counts = [len(c.chunk_text.split()) for c in chunks]
    topics = sorted({c.metadata.topic for c in chunks})
    if word_counts:
        st.caption(
            f"{len(chunks)} chunks · {sum(word_counts)} words "
            f"(avg {sum(word_counts) // len(chunks)}/chunk) · "
            f"topics: {', '.join(topics)}"
        )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    topics = get_vector_store().get_collection_stats()["topics"]
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_choice = st.selectbox("Topic", [ALL_FILTER, *topics], key="topic_select")
    with col_diff:
        difficulty_choice = st.selectbox(
            "Difficulty", DIFFICULTY_OPTIONS, key="difficulty_select"
        )
    st.session_state.topic_filter = None if topic_choice == ALL_FILTER else topic_choice
    st.session_state.difficulty_filter = (
        None if difficulty_choice == ALL_FILTER else difficulty_choice
    )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask about a deep learning topic...")
    if not query or not query.strip():
        return

    st.session_state.chat_history.append({"role": "user", "content": query})

    # Per-turn fields are reset explicitly: the checkpointer keeps state
    # between turns, so stale results from the last question must not leak in.
    graph_input = {
        "messages": [HumanMessage(content=query)],
        "topic_filter": st.session_state.topic_filter,
        "difficulty_filter": st.session_state.difficulty_filter,
        "retrieved_chunks": [],
        "no_context_found": False,
        "final_response": None,
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        with st.spinner("Thinking..."):
            result = graph.invoke(graph_input, config=config)
    except Exception as exc:  # LLM/API/connection failures should not crash the UI
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"}
        )
        st.rerun()

    response: AgentResponse | None = result.get("final_response")
    if response is None:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "⚠️ The agent returned no response."}
        )
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "no_context_found": response.no_context_found,
            }
        )

    # Remember which chunks were used so the viewer can highlight them.
    st.session_state.last_retrieved_scores = {
        chunk.chunk_id: chunk.score for chunk in result.get("retrieved_chunks", [])
    }
    st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
