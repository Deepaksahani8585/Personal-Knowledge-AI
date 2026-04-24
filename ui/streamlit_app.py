from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.config import settings
from app.ingest import DocumentIngestor
from app.main import KnowledgeAssistant


st.set_page_config(
    page_title="Personal Knowledge AI",
    layout="wide",
)

st.title("📚 Personal Knowledge AI Assistant")
st.caption("Answers only from your local documents using Ollama")

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


with st.sidebar:
    st.header("Settings")

    folder_path = st.text_input(
        "Documents folder path",
        value=os.path.abspath(settings.data_dir),
    )

    explicit_language = st.selectbox(
        "Response language",
        options=["auto", "hinglish", "hindi", "english"],
        index=0,
    )

    st.markdown("---")
    st.markdown(f"**Ollama Model:** `{settings.ollama_model}`")

    force_reingest = st.checkbox(
        "Force re-ingest",
        value=False,
        help="Only turn this ON when you add new books/files or change chunk settings.",
    )

    if st.button("Process & Ingest Documents"):
        try:
            with st.spinner("Processing documents... please wait"):
                ingestor = DocumentIngestor()
                result = ingestor.run(
                    folder_path=folder_path,
                    force=force_reingest,
                )

            st.session_state.assistant = None

            if result["status"] == "skipped":
                st.info(result["message"])
            else:
                st.success(
                    f"Done. Parsed {result['parsed_units']} units and created "
                    f"{result['chunks']} chunks."
                )

        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")


query = st.text_area(
    "Ask your question",
    height=140,
    placeholder="Example: Why is line profiling more informative than built-in profiling?",
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_clicked = st.button("Get Answer")

with col2:
    clear_clicked = st.button("Clear")


if clear_clicked:
    st.session_state.last_result = None
    st.rerun()


if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            if st.session_state.assistant is None:
                with st.spinner("Loading assistant..."):
                    st.session_state.assistant = KnowledgeAssistant()

            lang = None if explicit_language == "auto" else explicit_language

            with st.spinner("Generating answer from your documents..."):
                result = st.session_state.assistant.ask(
                    query,
                    explicit_language=lang,
                )

            st.session_state.last_result = result

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("Make sure you have ingested documents first and Ollama is running.")


result = st.session_state.last_result

if result:
    st.subheader("Answer")
    st.write(result.get("answer", ""))

    st.subheader("Search Stats")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric(
            "Best Relevance",
            f"{result.get('best_relevance_percent', 0)}%",
        )

    with col_b:
        st.metric(
            "Average Relevance",
            f"{result.get('avg_relevance_percent', 0)}%",
        )

    with col_c:
        st.metric(
            "Retrieval Time",
            f"{result.get('retrieval_time', 0)} sec",
        )

    with col_d:
        st.metric(
            "Total Time",
            f"{result.get('total_time', 0)} sec",
        )

    st.caption(
        "Relevance % means how closely retrieved document chunks matched your question. "
        "It is not guaranteed answer accuracy."
    )

    st.subheader("Sources")
    sources = result.get("sources", [])

    if sources:
        for source in sources:
            st.markdown(f"- {source}")
    else:
        st.info("No sources found.")

    with st.expander("Retrieved Chunks"):
        chunks = result.get("retrieved_chunks", [])

        if not chunks:
            st.info("No chunks retrieved.")
        else:
            for idx, chunk in enumerate(chunks, start=1):
                score = chunk.get("score", 0)
                score_percent = round(max(0, min(score * 100, 100)), 2)

                st.markdown(f"### Chunk {idx}")
                st.markdown(f"- **File:** {chunk.get('file_name')}")
                st.markdown(f"- **Page:** {chunk.get('page')}")
                st.markdown(f"- **Section:** {chunk.get('section')}")
                st.markdown(f"- **Score:** {score:.4f}")
                st.markdown(f"- **Relevance:** {score_percent}%")
                st.text(chunk.get("text", "")[:2500])
                st.divider()