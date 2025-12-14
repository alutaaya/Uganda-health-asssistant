# app.py
# Streamlit RAG (TEXT + TABLES, NO OCR)
# Stable, conversational, context-guarded
# Streamlit Community Cloud compatible

import os
import streamlit as st
import nest_asyncio

# -------------------------------------------------
# STREAMLIT PAGE CONFIG (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="Uganda Health Care Assistant",
    page_icon="🩺",
    layout="wide",
)

nest_asyncio.apply()

from pypdf import PdfReader
import pdfplumber

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document as LIDocument
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# -------------------------------------------------
# CONFIG (LOCAL + STREAMLIT CLOUD SAFE)
# -------------------------------------------------
OPENAI_API_KEY = None

# 1. Try Streamlit Secrets (Cloud)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# 2. Fallback to environment variable (local dev)
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 3. Stop if missing
if not OPENAI_API_KEY:
    st.error(
        "❌ OpenAI API key not found.\n\n"
        "Streamlit Cloud:\n"
        "  App → Settings → Secrets → OPENAI_API_KEY\n\n"
        "Local run:\n"
        "  set OPENAI_API_KEY=sk-..."
    )
    st.stop()

PDF_FOLDER = "pdfs"
PERSIST_DIR = "storage"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
STRICT_QA_PROMPT = PromptTemplate(
    """You are a clinical guideline assistant.

Answer using ONLY the context below.
If the context does not contain the answer, respond exactly with:
I do not know.

Rules:
- Do NOT ask for clarification
- Do NOT use outside knowledge

Context:
{context_str}

Question: {query_str}

Answer:"""
)

# -------------------------------------------------
# MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    Settings.llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )
    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )

# -------------------------------------------------
# PDF TEXT + TABLE EXTRACTION
# -------------------------------------------------
def extract_text_and_tables(pdf_path: str) -> list[LIDocument]:
    docs = []
    fname = os.path.basename(pdf_path)

    # ---- TEXT
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    LIDocument(
                        text=text,
                        metadata={
                            "source_file": fname,
                            "page": i + 1,
                        },
                    )
                )
    except Exception:
        pass

    # ---- TABLES
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables() or []
                for table in tables:
                    rows = []
                    for row in table:
                        clean = [c.strip() if c else "" for c in row]
                        rows.append(" | ".join(clean))
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        docs.append(
                            LIDocument(
                                text=f"Table content:\n{table_text}",
                                metadata={
                                    "source_file": fname,
                                    "page": i + 1,
                                    "type": "table",
                                },
                            )
                        )
    except Exception:
        pass

    return docs

# -------------------------------------------------
# INDEX
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_or_load_index():
    load_models()

    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage)

    all_docs = []
    for fn in os.listdir(PDF_FOLDER):
        if fn.lower().endswith(".pdf"):
            all_docs.extend(
                extract_text_and_tables(os.path.join(PDF_FOLDER, fn))
            )

    if not all_docs:
        st.error("No usable text or tables found in PDFs.")
        st.stop()

    parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=150,
    )
    nodes = parser.get_nodes_from_documents(all_docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index

# -------------------------------------------------
# ANSWERING (GUARDED)
# -------------------------------------------------
def guarded_answer(index, query, memory, top_k=10, threshold=0.30):
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved = retriever.retrieve(query)

    if not retrieved:
        return "I do not know.", []

    best_score = max(r.score or 0 for r in retrieved)
    if best_score < threshold:
        return "I do not know.", retrieved

    qe = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=STRICT_QA_PROMPT,
        memory=memory,
    )

    return str(qe.query(query)), retrieved

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📚 Knowledge Base")

    pdfs = sorted(
        f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")
    )
    if pdfs:
        st.selectbox(
            "Available PDFs (context)",
            options=pdfs,
            index=0,
        )
        st.caption(f"{len(pdfs)} PDFs indexed")
    else:
        st.warning("No PDFs found")

    st.divider()

    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000
        )
        st.rerun()

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🩺 Uganda Health Care Assistant")
st.caption("Answers are grounded strictly in guideline text and tables.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000
    )

with st.spinner("Loading knowledge base…"):
    index = build_or_load_index()

with st.expander("🔍 Index diagnostics"):
    st.write("Indexed chunks:", len(index.docstore.docs))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a clinical or programmatic question")

if user_q:
    st.session_state.messages.append(
        {"role": "user", "content": user_q}
    )

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = guarded_answer(
                index=index,
                query=user_q,
                memory=st.session_state.memory,
            )

        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        if sources:
            with st.expander("Sources"):
                shown = set()
                for r in sources:
                    md = r.node.metadata
                    src = (
                        md.get("source_file"),
                        md.get("page"),
                    )
                    if src not in shown:
                        shown.add(src)
                        st.write(f"- {src[0]}, page {src[1]}")
