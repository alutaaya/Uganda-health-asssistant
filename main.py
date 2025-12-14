# app.py
# Streamlit RAG – Streamlit Cloud HEALTH-CHECK SAFE

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

# -------------------------------------------------
# BASIC CONFIG (LIGHTWEIGHT)
# -------------------------------------------------
PDF_FOLDER = "pdfs"
PERSIST_DIR = "storage"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------------------------------
# API KEY (SAFE)
# -------------------------------------------------
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key missing in Streamlit Secrets.")
    st.stop()

# -------------------------------------------------
# GOOGLE DRIVE PDF SOURCES
# -------------------------------------------------
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
}

# -------------------------------------------------
# LIGHTWEIGHT UI (HEALTH CHECK PASSES HERE)
# -------------------------------------------------
st.title("🩺 Uganda Health Care Assistant")
st.caption("Initializing knowledge base…")

# -------------------------------------------------
# DEFERRED FUNCTIONS (HEAVY IMPORTS INSIDE)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_pdfs_present():
    for name, fid in PDF_FILES.items():
        path = os.path.join(PDF_FOLDER, name)
        if os.path.exists(path):
            continue
        url = f"https://drive.google.com/uc?export=download&id={fid}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

@st.cache_resource(show_spinner=False)
def load_index():
    # ⬇️ HEAVY IMPORTS ONLY NOW
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
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )
    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )

    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage)

    docs = []
    for fn in os.listdir(PDF_FOLDER):
        if not fn.endswith(".pdf"):
            continue
        path = os.path.join(PDF_FOLDER, fn)

        reader = PdfReader(path)
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                docs.append(
                    LIDocument(
                        text=text,
                        metadata={"source_file": fn, "page": i + 1},
                    )
                )

    parser = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=120)
    nodes = parser.get_nodes_from_documents(docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

# -------------------------------------------------
# SAFE EXECUTION AFTER HEALTH CHECK
# -------------------------------------------------
if "index" not in st.session_state:
    with st.spinner("Preparing knowledge base (first run)…"):
        ensure_pdfs_present()
        st.session_state.index = load_index()

index = st.session_state.index

st.success("Knowledge base ready ✅")
st.info("You can now ask questions.")
