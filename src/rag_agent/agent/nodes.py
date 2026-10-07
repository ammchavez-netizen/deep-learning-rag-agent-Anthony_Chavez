"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from loguru import logger

from rag_agent.agent.prompts import (
    QUERY_REWRITE_PROMPT,
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager

# NOTE: LangGraph passes nodes a plain dict (AgentState is a TypedDict), so
# state is read with state.get("field") / state["field"], never state.field.

# Rewritten queries longer than this are assumed to be the model rambling
# rather than producing keywords, and are discarded.
MAX_REWRITTEN_QUERY_CHARS = 300
CHARS_PER_TOKEN = 4  # rough estimate used for history trimming


@lru_cache(maxsize=1)
def _get_store() -> VectorStoreManager:
    """Return one shared VectorStoreManager (loading the embedding model is slow)."""
    return VectorStoreManager()


def _latest_human_message(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent HumanMessage, or '' if none."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _approx_token_count(messages: list[BaseMessage]) -> int:
    """Estimate tokens in a message list (no tokenizer download required)."""
    return sum(len(str(m.content)) // CHARS_PER_TOKEN + 4 for m in messages)


def _no_context_update(rewritten_query: str) -> dict:
    """
    Build the state update for the hallucination guard.

    Shared by retrieval_node (so the graph can end right after a failed
    retrieval) and generation_node (so the guard holds even if generation
    is reached some other way).
    """
    message = (
        "I was unable to find relevant information in the corpus for your query. "
        "This may mean the topic is not yet covered in the study material, or "
        "your query may need to be rephrased. Please try a more specific "
        "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
    )
    response = AgentResponse(
        answer=message,
        sources=[],
        confidence=0.0,
        no_context_found=True,
        rewritten_query=rewritten_query,
    )
    return {"final_response": response, "messages": [AIMessage(content=message)]}


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    original_query = _latest_human_message(state.get("messages", []))

    if not original_query.strip():
        return {"original_query": original_query, "rewritten_query": original_query}

    try:
        llm = LLMFactory(get_settings()).create()
        reply = llm.invoke(QUERY_REWRITE_PROMPT.format(original_query=original_query))
        # Keep the first line only and strip quotes the model may add.
        lines = str(reply.content).strip().splitlines()
        rewritten = lines[0].strip().strip("\"'`") if lines else ""
        if not rewritten or len(rewritten) > MAX_REWRITTEN_QUERY_CHARS:
            rewritten = original_query
    except Exception as exc:
        # Rewriting is an optimisation — never let it break the pipeline.
        logger.warning("Query rewrite failed ({}); using original query", exc)
        rewritten = original_query

    logger.debug("Rewrote '{}' -> '{}'", original_query[:60], rewritten[:60])
    return {"original_query": original_query, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    store = _get_store()
    rewritten = state.get("rewritten_query") or ""
    original = state.get("original_query") or ""
    filters = {
        "topic_filter": state.get("topic_filter"),
        "difficulty_filter": state.get("difficulty_filter"),
    }

    chunks = store.query(query_text=rewritten, **filters)

    # One retry: if the keyword-style rewrite found nothing, fall back to the
    # user's own wording before concluding the corpus has no answer.
    if not chunks and original and original != rewritten:
        logger.debug("No hits for rewritten query; retrying with original query")
        chunks = store.query(query_text=original, **filters)

    if not chunks:
        # Hallucination guard: build the "no context" response right here so
        # the graph can end without ever calling the LLM.
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
            **_no_context_update(rewritten),
        }
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()

    # ---- Hallucination Guard -----------------------------------------------
    # Checked before the LLM is created: the guard needs no model, so it must
    # still work when credentials are missing or the server is down.
    retrieved_chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    if state.get("no_context_found") or not retrieved_chunks:
        return _no_context_update(state.get("rewritten_query") or "")

    llm = LLMFactory(settings).create()

    # ---- Build Context from Retrieved Chunks --------------------------------
    # 1. Context string: one cited block per retrieved chunk.
    context = "\n".join(
        f"[SOURCE: {c.metadata.topic} | {c.metadata.source}]\n{c.chunk_text}\n"
        for c in retrieved_chunks
    )

    # 2. Confidence = average similarity of the chunks we are answering from.
    confidence = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)

    # 3. Prompt. The context is folded into the single system message because
    #    some local chat templates (LM Studio / Ollama models) reject a second
    #    system message.
    system_message = SystemMessage(
        content=f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
    )
    messages: list[BaseMessage] = state.get("messages", [])
    original_query = state.get("original_query") or _latest_human_message(messages)
    question = HumanMessage(content=original_query)

    # History = everything before the current question, trimmed to whatever
    # token budget is left once the system prompt, context and question fit.
    ends_with_question = bool(messages) and isinstance(messages[-1], HumanMessage)
    history = messages[:-1] if ends_with_question else messages
    used = _approx_token_count([system_message, question])
    budget = settings.max_context_tokens - used
    trimmed_history = (
        trim_messages(
            history,
            max_tokens=budget,
            strategy="last",
            token_counter=_approx_token_count,
            start_on="human",
            allow_partial=False,
        )
        if history and budget > 0
        else []
    )

    # 4. Generate.
    reply = llm.invoke([system_message, *trimmed_history, question])
    answer = str(reply.content).strip()

    # 5. Structured response; citations de-duplicated, order preserved.
    sources = list(dict.fromkeys(c.to_citation() for c in retrieved_chunks))
    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query") or "",
    )

    # 6-7. The reducer appends this AIMessage to the conversation history.
    return {"final_response": response, "messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    # retrieval_node already retried once (rewritten query, then the original
    # wording) and, on failure, already built the no-context response — so
    # ending here is safe and skips an unnecessary LLM call.
    return "end" if state.get("no_context_found") else "generate"
