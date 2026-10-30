"""Chat UI that talks to the FastAPI backend.
Run the API first, then: streamlit run ui/streamlit_app.py"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuChat", page_icon="📄")
st.title("📄 DocuChat — chat with your documents")

with st.sidebar:
    st.header("Upload documents")
    files = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index files", disabled=not files):
        for f in files:
            r = requests.post(f"{API_URL}/documents", files={"file": (f.name, f.getvalue())}, timeout=300)
            if r.ok:
                st.success(f"{f.name}: {r.json()['chunks_added']} chunks added")
            else:
                st.error(f"{f.name}: {r.text}")
    try:
        st.caption(f"Chunks indexed: {requests.get(f'{API_URL}/health', timeout=5).json()['chunks_indexed']}")
    except requests.RequestException:
        st.warning("API not reachable. Start it with `uvicorn app.api:app`.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    page = f", p.{s['page']}" if s["page"] else ""
                    st.markdown(f"**[{s['n']}] {s['source']}{page}** (score {s['score']})\n\n> {s['text']}…")

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.history.append(("user", question, None))
    with st.spinner("Thinking…"):
        try:
            r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            r.raise_for_status()
            data = r.json()
            st.session_state.history.append(("assistant", data["answer"], data["sources"]))
        except requests.RequestException as e:
            st.session_state.history.append(("assistant", f"Error: {e}", None))
    st.rerun()
