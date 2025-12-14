# app.py
# Streamlit RAG (TEXT + TABLES, NO OCR)
# Cloud-safe startup with delayed indexing
# Prevents Streamlit health-check failures

import os
import streamlit as st
import nest_asyncio
import requests

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
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ OpenAI API key not found.\n\n"
        "Streamlit Cloud:\n"
        "  App → Settings → Secrets → OPENAI_API_KEY\n\n"
        "Local:\n"
        "  set OPENAI_API_KEY=sk-..."
    )
    st.stop()

PDF_FOLDER = "pdfs"
PERSIST_DIR = "storage"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------------------------------
# GOOGLE DRIVE PDF SOURCES
# -------------------------------------------------
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
    "UCG.pdf": "1f68UdsRdYwXW5DNN61pBNQXK7TkpMc0o",
    "PMDT.pdf": "1zhFrJC90olY7aaledw_RyKR_sC58XV2j",
    "HTS.pdf": "1mI8r0B2GmRGoWrJAEAOBZcpXb5znPmWs",
    "IPC.pdf": "1DKmCrueBly6jFtUP9Ox631jqzGDsR2tV",
    "TB_children.pdf": "1HUtgNMO_D-CK6ofLPf6egteHG7lhsd5S",
    "prevention.pdf": "1yTZ6JiB4ky8CcGK9tabkH3kLWCT2js4J",
    "TB_Lep.pdf": "1UUKe1PPgti_Gm6RgDq-kBexv2BgYXxTF",
    "CKD.pdf": "1sOVGB7R1IEu3kWQrdd0IZCNmxXHJ3jWC",
    "DSD.pdf": "1WRerkPmfRAzgPS234yP56aJ8zYXPwcjT",
}

# -------------------------------------------------
# DOWNLOAD PDFs (CACHED, SAFE)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_pdfs_present():
    for filename, file_id in PDF_FILES.items():
        path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(path):
            continue

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)

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
# PDF EXTRACTION
# -------------------------------------------------
def extract_text_and_tables(pdf_path: str) -> list[LIDocument]:
    docs = []
    fname = os.path.basename(pdf_path)

    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    LIDocument(
                        text=text,
                        metadata={"source_file": fname, "page": i + 1},
                    )
                )
    except Exception:
        pass

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join(c.strip() if c else "" for c in row)
                        for row in table
                    ]
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
# INDEX (CACHED ONCE)
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

    parser = SimpleNodeParser.from_defaults(
        chunk_size=800,
        chunk_overlap=120,
    )
    nodes = parser.get_nodes_from_documents(all_docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index

# -------------------------------------------------
# ANSWERING
# -------------------------------------------------
def guarded_answer(index, query, memory):
    retriever = index.as_retriever(similarity_top_k=5)
    retrieved = retriever.retrieve(query)

    if not retrieved:
        return "I do not know.", []

    qe = index.as_query_engine(
        text_qa_template=STRICT_QA_PROMPT,
        memory=memory,
    )
    return str(qe.query(query)), retrieved

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption("Guidelines loaded from Google Drive")

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

# -------------------------------------------------
# DELAY HEAVY WORK (CRITICAL FIX)
# -------------------------------------------------
if "index" not in st.session_state:
    with st.spinner("Preparing knowledge base (first run only)…"):
        st.write("Downloading guideline PDFs…")
        ensure_pdfs_present()

        st.write("Building search index…")
        st.session_state.index = build_or_load_index()

index = st.session_state.index

# -------------------------------------------------
# CHAT UI
# -------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a clinical or programmatic question")

if user_q:
    st.session_state.messages.append(
        {"role": "user", "content": user_q}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = guarded_answer(
                index, user_q, st.session_state.memory
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
                    src = (md.get("source_file"), md.get("page"))
                    if src not in shown:
                        shown.add(src)
                        st.write(f"- {src[0]}, page {src[1]}")
