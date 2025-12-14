import os
import time
import json
import threading
from typing import Dict, List, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Uganda Health Care Assistant"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    # Cloud Run: set via Secret Manager/env var. Local: export OPENAI_API_KEY=...
    print("WARNING: OPENAI_API_KEY is not set. The app will refuse queries until it is set.")

# Cloud Run ephemeral disk: use /tmp
PDF_FOLDER = os.environ.get("PDF_FOLDER", "/tmp/pdfs")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/tmp/storage")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Your Google Drive file IDs
PDF_FILES: Dict[str, str] = {
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

# Guardrails
TOP_K = int(os.environ.get("TOP_K", "8"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.30"))

# -----------------------------
# Simple in-memory app state
# -----------------------------
class AppState:
    ready: bool = False
    building: bool = False
    error: Optional[str] = None
    index = None  # LlamaIndex VectorStoreIndex
    build_started_at: Optional[float] = None

STATE = AppState()
STATE_LOCK = threading.Lock()

# -----------------------------
# UI (single-page)
# -----------------------------
INDEX_HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --accent: #7c5cff;
      --border: rgba(255,255,255,0.12);
      --shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: radial-gradient(1200px 800px at 10% 10%, rgba(124,92,255,0.25), transparent 60%),
                  radial-gradient(900px 600px at 85% 35%, rgba(0,200,255,0.18), transparent 60%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: stretch;
      justify-content: center;
    }}
    .wrap {{
      width: min(1100px, 100%);
      padding: 24px 16px 40px;
    }}
    .header {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .badge {{
      width: 44px; height: 44px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(124,92,255,0.9), rgba(0,200,255,0.65));
      display: grid; place-items: center;
      box-shadow: var(--shadow);
      font-size: 22px;
    }}
    h1 {{
      font-size: 22px;
      margin: 0;
      letter-spacing: 0.2px;
    }}
    .sub {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .status {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      background: rgba(255,255,255,0.03);
    }}
    .status small {{
      color: var(--muted);
    }}
    .pill {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(124,92,255,0.18);
      border: 1px solid rgba(124,92,255,0.35);
      color: rgba(255,255,255,0.9);
      font-size: 12px;
      white-space: nowrap;
    }}
    .chat {{
      height: 62vh;
      overflow: auto;
      padding: 16px;
      scroll-behavior: smooth;
    }}
    .msg {{
      display: flex;
      gap: 10px;
      margin: 12px 0;
      align-items: flex-start;
    }}
    .avatar {{
      width: 34px; height: 34px;
      border-radius: 12px;
      background: var(--panel2);
      border: 1px solid var(--border);
      display: grid; place-items: center;
      flex: 0 0 auto;
      color: rgba(255,255,255,0.85);
      font-size: 14px;
    }}
    .bubble {{
      max-width: 85%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      line-height: 1.35;
      font-size: 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .user .bubble {{
      background: rgba(124,92,255,0.14);
      border-color: rgba(124,92,255,0.35);
    }}
    .sources {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed rgba(255,255,255,0.18);
      color: var(--muted);
      font-size: 12px;
    }}
    .composer {{
      display: flex;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--border);
      background: rgba(0,0,0,0.10);
    }}
    textarea {{
      flex: 1 1 auto;
      resize: none;
      height: 44px;
      padding: 12px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      outline: none;
      font-size: 14px;
    }}
    button {{
      flex: 0 0 auto;
      padding: 0 16px;
      height: 44px;
      border-radius: 12px;
      border: 1px solid rgba(124,92,255,0.45);
      background: linear-gradient(135deg, rgba(124,92,255,0.95), rgba(0,200,255,0.55));
      color: white;
      font-weight: 600;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
    }}
    .row {{
      display:flex; gap:10px; align-items:center;
    }}
    .link {{
      color: rgba(255,255,255,0.85);
      text-decoration: none;
      border-bottom: 1px dotted rgba(255,255,255,0.3);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="badge">🩺</div>
      <div>
        <h1>{APP_TITLE}</h1>
        <div class="sub">Answers are grounded strictly in guideline text and tables.</div>
      </div>
    </div>

    <div class="card">
      <div class="status">
        <div class="row">
          <div id="statusText"><small>Checking system status…</small></div>
          <div id="pill" class="pill">Starting</div>
        </div>
        <div class="row">
          <a class="link" href="/health" target="_blank">Health</a>
          <a class="link" href="/api/status" target="_blank">Status JSON</a>
        </div>
      </div>

      <div id="chat" class="chat"></div>

      <div class="composer">
        <textarea id="q" placeholder="Ask a clinical or programmatic question…"></textarea>
        <button id="send">Send</button>
      </div>
    </div>
  </div>

<script>
  const chat = document.getElementById("chat");
  const q = document.getElementById("q");
  const send = document.getElementById("send");
  const statusText = document.getElementById("statusText");
  const pill = document.getElementById("pill");

  function addMessage(role, text, sources) {{
    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "user" : "assistant");

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    if (sources && sources.length) {{
      const s = document.createElement("div");
      s.className = "sources";
      s.textContent = "Sources: " + sources.join(", ");
      bubble.appendChild(s);
    }}

    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }}

  async function refreshStatus() {{
    try {{
      const r = await fetch("/api/status");
      const data = await r.json();
      if (data.ready) {{
        statusText.innerHTML = "<small>Ready</small>";
        pill.textContent = "Ready";
      }} else if (data.building) {{
        statusText.innerHTML = "<small>Preparing knowledge base…</small>";
        pill.textContent = "Warming up";
      }} else if (data.error) {{
        statusText.innerHTML = "<small>Error: " + data.error + "</small>";
        pill.textContent = "Error";
      }} else {{
        statusText.innerHTML = "<small>Starting…</small>";
        pill.textContent = "Starting";
      }}
      send.disabled = !data.ready;
    }} catch (e) {{
      statusText.innerHTML = "<small>Unable to reach server status</small>";
      pill.textContent = "Offline";
      send.disabled = true;
    }}
  }}

  async function ask() {{
    const text = q.value.trim();
    if (!text) return;
    addMessage("user", text);
    q.value = "";
    send.disabled = true;

    try {{
      const r = await fetch("/api/ask", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ question: text }})
      }});
      const data = await r.json();
      if (!r.ok) {{
        addMessage("assistant", data.detail || "Server error.");
      }} else {{
        addMessage("assistant", data.answer || "", data.sources || []);
      }}
    }} catch (e) {{
      addMessage("assistant", "Network error. Please try again.");
    }} finally {{
      await refreshStatus();
    }}
  }}

  send.addEventListener("click", ask);
  q.addEventListener("keydown", (e) => {{
    if (e.key === "Enter" && !e.shiftKey) {{
      e.preventDefault();
      ask();
    }}
  }});

  refreshStatus();
  setInterval(refreshStatus, 5000);
</script>
</body>
</html>
"""

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


class AskBody(BaseModel):
    question: str


# -----------------------------
# Google Drive downloader (handles confirm token)
# -----------------------------
def _gdrive_download(file_id: str, dest_path: str, timeout: int = 120) -> None:
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}

    resp = session.get(url, params=params, stream=True, timeout=timeout)
    resp.raise_for_status()

    # If Google shows virus scan warning, it sets a confirm token cookie
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        params["confirm"] = token
        resp = session.get(url, params=params, stream=True, timeout=timeout)
        resp.raise_for_status()

    # Write atomically to avoid partial reads
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, dest_path)


def ensure_pdfs_present() -> None:
    for fname, fid in PDF_FILES.items():
        path = os.path.join(PDF_FOLDER, fname)
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            continue
        _gdrive_download(fid, path)


# -----------------------------
# LlamaIndex build/load
# -----------------------------
def build_or_load_index():
    # Heavy imports here (Docker/Cloud Run is fine)
    from pypdf import PdfReader
    import pdfplumber

    from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import Document
    from llama_index.core.prompts import PromptTemplate
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

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

    Settings.llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # Load if persisted
    if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage), STRICT_QA_PROMPT

    docs: List[Document] = []

    for fn in os.listdir(PDF_FOLDER):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_FOLDER, fn)

        # Text
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(text=text, metadata={"source_file": fn, "page": i + 1}))
        except Exception:
            pass

        # Tables (best-effort)
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for table in page.extract_tables() or []:
                        rows = [
                            " | ".join((c.strip() if c else "") for c in row)
                            for row in table
                        ]
                        table_text = "\n".join(rows).strip()
                        if table_text:
                            docs.append(
                                Document(
                                    text=f"Table content:\n{table_text}",
                                    metadata={"source_file": fn, "page": i + 1, "type": "table"},
                                )
                            )
        except Exception:
            pass

    parser = SimpleNodeParser.from_defaults(chunk_size=900, chunk_overlap=120)
    nodes = parser.get_nodes_from_documents(docs)

    storage = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index, STRICT_QA_PROMPT


def start_background_build():
    with STATE_LOCK:
        if STATE.ready or STATE.building:
            return
        STATE.building = True
        STATE.build_started_at = time.time()
        STATE.error = None

    def _worker():
        try:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set.")

            ensure_pdfs_present()
            idx, prompt = build_or_load_index()

            with STATE_LOCK:
                STATE.index = (idx, prompt)
                STATE.ready = True
                STATE.building = False
                STATE.error = None
        except Exception as e:
            with STATE_LOCK:
                STATE.error = str(e)
                STATE.ready = False
                STATE.building = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


@app.on_event("startup")
def on_startup():
    # Kick off build in the background so the service becomes reachable fast
    start_background_build()


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    with STATE_LOCK:
        return {
            "ok": True,
            "ready": STATE.ready,
            "building": STATE.building,
            "error": STATE.error,
        }


@app.get("/api/status")
def api_status():
    with STATE_LOCK:
        return {
            "ready": STATE.ready,
            "building": STATE.building,
            "error": STATE.error,
            "pdf_count": len([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]),
        }


@app.post("/api/ask")
def api_ask(body: AskBody):
    with STATE_LOCK:
        if STATE.error:
            return JSONResponse(status_code=500, content={"detail": STATE.error})
        if not STATE.ready or not STATE.index:
            return JSONResponse(status_code=503, content={"detail": "System is warming up. Please retry in a moment."})

        idx, prompt = STATE.index

    # Build guarded answer
    from llama_index.core.prompts import PromptTemplate  # light import
    query = body.question.strip()
    if not query:
        return JSONResponse(status_code=400, content={"detail": "Question is empty."})

    retriever = idx.as_retriever(similarity_top_k=TOP_K)
    retrieved = retriever.retrieve(query)

    if not retrieved:
        return {"answer": "I do not know.", "sources": []}

    best_score = max((r.score or 0.0) for r in retrieved)
    if best_score < THRESHOLD:
        sources = _format_sources(retrieved)
        return {"answer": "I do not know.", "sources": sources}

    qe = idx.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=prompt,
    )

    answer = str(qe.query(query))
    sources = _format_sources(retrieved)
    return {"answer": answer, "sources": sources}


def _format_sources(retrieved) -> List[str]:
    shown = set()
    out = []
    for r in retrieved:
        md = getattr(r.node, "metadata", {}) or {}
        src = md.get("source_file")
        page = md.get("page")
        key = (src, page)
        if src and page and key not in shown:
            shown.add(key)
            out.append(f"{src} p.{page}")
    return out
