import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import BENCHMARK_QUERIES, parse_markdown, TfidfEmbedder
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker, HeadingChunker, compute_similarity as cosine_sim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

files = sorted(Path("data/thu-vien").glob("*.md"))
q1 = BENCHMARK_QUERIES[0]
print("=== THỬ NGHIỆM A/B: CÂU 1 (Q1) ===")
print(f"Query: {q1['query']}")
print(f"Tài liệu đúng: {q1['expected_docs']}")

strategies = [
    ("FixedSize", FixedSizeChunker(300, 50)),
    ("Sentence", SentenceChunker(3)),
    ("Recursive", RecursiveChunker(chunk_size=300)),
    ("Heading", HeadingChunker(300)),
]

for name, chunker in strategies:
    records = []
    for fp in files:
        fm, body = parse_markdown(fp)
        did = fm.get("doc_id", fp.stem)
        for i, c in enumerate(chunker.chunk(body)):
            records.append({
                "id": f"{did}#{i}",
                "doc_id": did,
                "content": c,
                "metadata": {**fm, "doc_id": did}
            })

    emb = TfidfEmbedder()
    emb.fit([r["content"] for r in records])
    for r in records:
        r["embedding"] = emb.encode(r["content"])
    q_emb = emb.encode(q1["query"])

    # 1. Có filter
    cands_filtered = [r for r in records if r["metadata"].get("audience") == "student"]
    scored_filtered = []
    for r in cands_filtered:
        scored_filtered.append({
            "id": r["id"],
            "doc_id": r["doc_id"],
            "score": cosine_sim(q_emb, r["embedding"]),
            "audience": r["metadata"].get("audience")
        })
    scored_filtered.sort(key=lambda x: x["score"], reverse=True)

    # 2. Không filter
    scored_unfiltered = []
    for r in records:
        scored_unfiltered.append({
            "id": r["id"],
            "doc_id": r["doc_id"],
            "score": cosine_sim(q_emb, r["embedding"]),
            "audience": r["metadata"].get("audience")
        })
    scored_unfiltered.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n--- Chiến lược {name} ---")
    print("KHI CÓ FILTER {'audience': 'student'}:")
    for idx, r in enumerate(scored_filtered[:3], 1):
        target = "[DUNG]" if r["doc_id"] in q1["expected_docs"] else "[SAI]"
        print(f"  Top {idx}: {target} {r['id']} (score={r['score']:.3f}, audience={r['audience']})")

    print("KHI KHÔNG CÓ FILTER:")
    for idx, r in enumerate(scored_unfiltered[:3], 1):
        target = "[DUNG]" if r["doc_id"] in q1["expected_docs"] else "[SAI]"
        print(f"  Top {idx}: {target} {r['id']} (score={r['score']:.3f}, audience={r['audience']})")
