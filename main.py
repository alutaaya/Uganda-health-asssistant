# app.py
# Uganda Health Care Assistant
# Streamlit Community Cloud – STABLE VERSION

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
# BASIC PATH CONFIG
# -------------------------------------------------
PDF_FOLDER = "pdfs"
PERSIST_DIR = "storage"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------------------------------
# OPENAI API KEY (STREAMLIT CLOUD SAFE)
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
        "Streamlit Cloud → App → Settings → Secrets → OPENAI_API_KEY"
    )
    st.stop()

# -------------------------------------------------
# GOOGLE DRIVE PDF SOURCES
# -------------------------------------------------
PDF_FILES = {
    "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf": "1rY_UE-sIw4f5Z5VUt0pyllPs7tSENsSr",
    "PrEP.pdf": "1n0Mtds2dSb6lCaJm6Ic8-NtaEIHnH5UQ",
    "NTLP_manual.pdf": "1SEPZ9j5zew9XcIeCdrXwzcopCulf_APZ",
    "UCG.pdf": "1f68UdsRdYwXW5DNN61pBNQXK7TkpMc0o",
}

# -------------------------------------------------
# LIGHT UI (HEALTH CHECK PASSES HERE)
# -------------------------------------------------
st.title("🩺 Uganda Health Care Assistant")
st.caption("Initializing knowledge base…")

# -------------------------------------------------
# STEP 1: DOWNLOAD PDFs (NO CACHING, NO RACE)
# -------------------------------------------------
def ensure_pdfs_present():
    for fname, fid in PDF_FILES.items():
        path = os.path.join(PDF_FOLDER, fname)

        if os.path.exists(path):
            continue

        url = f"https://drive.google.com/uc?export=download&id={fid}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)

# -------------------------------------------------
# STEP 2: BUILD / LOAD INDEX (CACHED, READ-ONLY)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_index():
    # Heavy imports ONLY inside cached function
    from pypdf import PdfReader
    import pdfplumber

    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        Settings,
        load_index_from_storage,
    )
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import Document
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    # Configure models
    Settings.llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
    )
    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )

    # Load existing index if present
    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage)

    # Build new index
    docs = []

    for fn in os.listdir(PDF_FOLDER):
        if not fn.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_FOLDER, fn)

        # ---- TEXT
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        metadata={"source_file": fn, "page": i + 1},
                    )
                )

        # ---- TABLES (safe version)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join(c.strip() if c else "" for c in row)
                        for row in table
                    ]
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        docs.append(
                            Document(
                                text=f"Table:\n{table_text}",
                                metadata={"source_file": fn, "page": i + 1},
                            )
                        )

    parser = SimpleNodeParser.from_defaults(
        chunk_size=800,
        chunk_overlap=120,
    )
    nodes = parser.get_nodes_from_documents(docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index

# -------------------------------------------------
# CONTROLLED INITIALIZATION (NO RACE CONDITIONS)
# -------------------------------------------------
if "pdfs_ready" not in st.session_state:
    with st.spinner("Downloading guideline PDFs…"):
        ensure_pdfs_present()
        st.session_state.pdfs_ready = True

if "index" not in st.session_state:
    with st.spinner("Building search index…"):
        st.session_state.index = load_index()

index = st.session_state.index

st.success("Knowledge base ready ✅")

# -------------------------------------------------
# CHAT UI
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a clinical or programmatic question")

if user_q:
    st.session_state.messages.append(
        {"role": "user", "content": user_q}
    )

    with st.chat_message("assistant"):
        qe = index.as_query_engine()
        answer = str(qe.query(user_q))
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
