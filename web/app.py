import re
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench import BENCHMARK_QUERIES, TfidfEmbedder, parse_markdown
from src.agent import KnowledgeBaseAgent
from src.chunking import (
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.models import Document
from src.store import EmbeddingStore

app = FastAPI(title="Lab 07 RAG Demo - Nhóm Mãi Là Lốp Trưởng")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = PROJECT_ROOT / "data" / "thu-vien"

# Global cache for stores and documents
DOCUMENTS_CACHE = []
STORES_CACHE = {}
AGENTS_CACHE = {}

STRATEGIES_DEF = {
    "recursive": {
        "name": "Recursive Chunker",
        "author": "Phạm Quang Huy (2A202602900)",
        "chunker": lambda: RecursiveChunker(chunk_size=300),
        "desc": "Tách đệ quy đa cấp theo \\n\\n, \\n, dấu câu, kích thước <= 300.",
    },
    "fixed": {
        "name": "Fixed-Size Chunker",
        "author": "Tạ Hoàng Vinh (2A202602543)",
        "chunker": lambda: FixedSizeChunker(chunk_size=700, overlap=50),
        "desc": "Cửa sổ trượt 700 ký tự với overlap 50 ký tự gối đầu.",
    },
    "sentence": {
        "name": "Sentence Chunker",
        "author": "Ngô Đức Chung (2A202602985)",
        "chunker": lambda: SentenceChunker(max_sentences_per_chunk=3),
        "desc": "Tách theo dấu câu, gom cụm tối đa 3 câu trọn vẹn ngữ pháp.",
    },
    "heading": {
        "name": "Heading Chunker (K4-L3A)",
        "author": "Bùi Tiến Cường (2A202602539)",
        "chunker": lambda: HeadingChunker(max_chunk_size=350),
        "desc": "Tách theo #, ## và Điều khoản, đệ quy bảo toàn provenance ngữ cảnh cha.",
    },
}


def rag_llm(prompt: str) -> str:
    """Mock/Heuristic LLM synthesis function for demo purpose."""
    match_ctx = re.search(r"NGỮ CẢNH:\s*(.*?)\s*YÊU CẦU:", prompt, re.DOTALL)
    if not match_ctx:
        return "Không tìm thấy thông tin phù hợp trong ngữ cảnh được cung cấp."

    ctx_text = match_ctx.group(1).strip()
    blocks = re.findall(r"(\[\d+\]\s*\(Nguồn:[^\)]+\)\s*([\s\S]*?))(?=\[\d+\]|\Z)", ctx_text)
    if not blocks:
        return f"Dựa trên thông tin được tìm thấy:\n{ctx_text[:300]}..."

    answers = []
    for full_block, content in blocks[:2]:
        first_line = full_block.strip().split("\n")[0]
        lines = [ln.strip() for ln in content.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        summary = " ".join(lines[:3]) if lines else content[:200]
        answers.append(f"{first_line}\n👉 {summary}")

    return "Dựa trên các quy định được trích xuất từ kho tri thức thư viện:\n\n" + "\n\n".join(answers)


def init_rag_system():
    global DOCUMENTS_CACHE, STORES_CACHE, AGENTS_CACHE
    DOCUMENTS_CACHE = []
    files = sorted(DATA_DIR.glob("*.md"))
    raw_docs = []
    for fp in files:
        fm, body = parse_markdown(fp)
        did = fm.get("doc_id", fp.stem)
        doc_info = {
            "doc_id": did,
            "title": fm.get("title", did),
            "audience": fm.get("audience", "all"),
            "category": fm.get("category", "general"),
            "source_url": fm.get("source_url", ""),
            "chars": len(body),
            "content": body,
            "metadata": fm,
        }
        DOCUMENTS_CACHE.append(doc_info)
        raw_docs.append((did, body, fm))

    # Initialize each strategy store
    for key, info in STRATEGIES_DEF.items():
        chunker = info["chunker"]()
        all_chunks = []
        doc_records = []
        for did, body, fm in raw_docs:
            chunks = chunker.chunk(body)
            for idx, c in enumerate(chunks):
                cid = f"{did}#{idx}"
                all_chunks.append(c)
                doc_records.append((cid, did, c, {**fm, "doc_id": did}))

        # Fit TF-IDF embedder
        embedder = TfidfEmbedder()
        embedder.fit(all_chunks)

        # Build EmbeddingStore using TF-IDF embedder function
        store = EmbeddingStore(embedding_fn=embedder.encode)
        docs_to_add = [Document(id=cid, content=text, metadata=meta) for cid, did, text, meta in doc_records]
        store.add_documents(docs_to_add)

        STORES_CACHE[key] = {
            "store": store,
            "embedder": embedder,
            "chunk_count": len(all_chunks),
        }
        AGENTS_CACHE[key] = KnowledgeBaseAgent(store, llm_fn=rag_llm)


# Run initialization on startup
init_rag_system()


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str
    strategy: str = "recursive"
    audience_filter: Optional[str] = None
    top_k: int = 3


class ChunkCompareRequest(BaseModel):
    doc_id: Optional[str] = None
    custom_text: Optional[str] = None


@app.get("/api/info")
def get_system_info():
    return {
        "group_name": "Mãi Là Lốp Trưởng",
        "lab": "Lab 07: Embedding & Vector Store (K4-L3A)",
        "members": [
            {"name": "Tạ Hoàng Vinh", "mssv": "2A202602543", "role": "Report Lead, FixedSizeChunker (700, 50)"},
            {"name": "Ngô Đức Chung", "mssv": "2A202602985", "role": "SentenceChunker (3 câu)"},
            {"name": "Bùi Tiến Cường", "mssv": "2A202602539", "role": "HeadingChunker (350)"},
            {"name": "Phạm Quang Huy", "mssv": "2A202602900", "role": "RecursiveChunker (300)"},
        ],
        "strategies": {
            k: {
                "name": v["name"],
                "author": v["author"],
                "desc": v["desc"],
                "chunk_count": STORES_CACHE[k]["chunk_count"],
            }
            for k, v in STRATEGIES_DEF.items()
        },
        "benchmark_queries": BENCHMARK_QUERIES,
    }


@app.get("/api/documents")
def list_documents():
    return [
        {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "audience": d["audience"],
            "category": d["category"],
            "source_url": d["source_url"],
            "chars": d["chars"],
            "metadata": d["metadata"],
        }
        for d in DOCUMENTS_CACHE
    ]


@app.get("/api/document/{doc_id}")
def get_document_content(doc_id: str):
    for d in DOCUMENTS_CACHE:
        if d["doc_id"] == doc_id:
            return d
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/query")
def run_query(req: QueryRequest):
    strat = req.strategy.lower()
    if strat not in STORES_CACHE:
        strat = "recursive"

    store_entry = STORES_CACHE[strat]
    store: EmbeddingStore = store_entry["store"]
    embedder: TfidfEmbedder = store_entry["embedder"]
    agent: KnowledgeBaseAgent = AGENTS_CACHE[strat]

    query_vec = embedder.encode(req.query)

    # 1. Retrieval
    metadata_filter = {"audience": req.audience_filter} if req.audience_filter and req.audience_filter != "all" else None
    if metadata_filter:
        results = store.search_with_filter(req.query, metadata_filter=metadata_filter, top_k=req.top_k)
    else:
        results = store.search(req.query, top_k=req.top_k)

    # 2. Agent Answer
    agent_answer = agent.answer(req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "strategy": strat,
        "strategy_name": STRATEGIES_DEF[strat]["name"],
        "audience_filter": req.audience_filter,
        "results": results,
        "answer": agent_answer,
    }


@app.get("/api/ab_test")
def run_ab_test():
    """Thực nghiệm A/B Testing bắt buộc: Chạy câu 1 có vs không có filter trên 4 chiến lược."""
    q1 = BENCHMARK_QUERIES[0]
    results = {}

    for strat_key, strat_entry in STORES_CACHE.items():
        store = strat_entry["store"]

        # With filter
        with_filter = store.search_with_filter(q1["query"], metadata_filter={"audience": "student"}, top_k=3)

        # Without filter
        without_filter = store.search(q1["query"], top_k=3)

        results[strat_key] = {
            "name": STRATEGIES_DEF[strat_key]["name"],
            "author": STRATEGIES_DEF[strat_key]["author"],
            "with_filter": with_filter,
            "without_filter": without_filter,
        }

    return {
        "query": q1["query"],
        "expected_docs": q1["expected_docs"],
        "gold_answer": q1["gold_answer"],
        "strategies": results,
    }


@app.post("/api/chunk_compare")
def compare_chunks(req: ChunkCompareRequest):
    text_to_test = ""
    doc_title = "Văn bản thử nghiệm"
    if req.doc_id:
        for d in DOCUMENTS_CACHE:
            if d["doc_id"] == req.doc_id:
                text_to_test = d["content"]
                doc_title = d["title"]
                break
    if not text_to_test and req.custom_text:
        text_to_test = req.custom_text
        doc_title = "Văn bản tùy chỉnh nhập tay"

    if not text_to_test:
        # Default to first document
        text_to_test = DOCUMENTS_CACHE[0]["content"]
        doc_title = DOCUMENTS_CACHE[0]["title"]

    results = {}
    for key, info in STRATEGIES_DEF.items():
        chunker = info["chunker"]()
        chunks = chunker.chunk(text_to_test)
        avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        results[key] = {
            "name": info["name"],
            "author": info["author"],
            "count": len(chunks),
            "avg_length": round(avg_len, 1),
            "chunks": chunks[:10],  # Return up to 10 sample chunks
        }

    return {
        "doc_title": doc_title,
        "text_length": len(text_to_test),
        "results": results,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = PROJECT_ROOT / "web" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Web interface loading...</h1>"


if __name__ == "__main__":
    import uvicorn
    print("Khởi động máy chủ Web Demo RAG tại: http://127.0.0.1:8000")
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
