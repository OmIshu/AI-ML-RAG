"""Streamlit frontend. Run: streamlit run app.py  (backend must be running)"""
import requests
import streamlit as st

import api_client as api


def pages(h: dict) -> str:
    return str(h["page_start"]) if h["page_start"] == h["page_end"] else f"{h['page_start']}-{h['page_end']}"


def show_answer(turn: dict):
    st.write(turn["answer"])
    if not st.session_state.get("show_details"):
        return
    if turn["citations"]:
        st.caption("Sources: " + " | ".join(f"[{c['n']}] {c['filename']}, p.{pages(c)}" for c in turn["citations"]))
    with st.expander(f"Retrieved chunks ({len(turn['retrieved'])}) - {turn['timings']['total_ms'] / 1000:.1f}s"):
        for i, h in enumerate(turn["retrieved"], 1):
            score = h["rerank_score"] if h.get("rerank_score") is not None else h["dense_score"]
            st.markdown(f"**[{i}] {h['filename']} - p.{pages(h)}** (score {score:.2f})")
            st.write(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""))


st.set_page_config(page_title="RAG OCR Chatbot", layout="centered")
st.title("RAG OCR Chatbot")

try:
    docs = api.documents()
except requests.RequestException:
    st.error(f"Backend not reachable at {api.BASE_URL}. Start it first (uvicorn ragbot.api.main:app --port 8000).")
    st.stop()

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Add to knowledge base", type="primary"):
        for f in files:
            with st.spinner(f"Processing {f.name} - large PDFs can take a few minutes..."):
                job = api.ingest_and_wait(f.name, f.getvalue())
            (st.success if job["status"] == "done" else st.error)(f"{f.name}: {job['message'] or job['status']}")
        st.rerun()

    for d in docs:
        st.caption(f"{d['filename']} ({d['pages']} pages)")
    scope = st.selectbox("Search in", ["All documents"] + [d["filename"] for d in docs])
    st.toggle("Show sources & retrieved chunks", key="show_details", value=False)

st.session_state.setdefault("history", [])
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        show_answer(turn)

if q := st.chat_input("Ask a question about your documents", disabled=not docs):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                turn = {"q": q, **api.chat(q, None if scope == "All documents" else scope)}
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()
        show_answer(turn)
        st.session_state.history.append(turn)

if not docs:
    st.info("Upload at least one PDF in the sidebar to get started.")
