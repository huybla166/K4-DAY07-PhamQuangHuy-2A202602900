import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import ChunkingStrategyComparator
from bench import parse_markdown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

comparator = ChunkingStrategyComparator()
docs_to_test = [
    "data/thu-vien/huit-borrow-student.md",
    "data/thu-vien/library-borrow-home.md",
    "data/thu-vien/library-faq.md"
]

for doc_path in docs_to_test:
    p = Path(doc_path)
    fm, body = parse_markdown(p)
    res = comparator.compare(body)
    print(f"=== {p.name} (Length: {len(body)} chars) ===")
    for strat, data in res.items():
        print(f"  - {strat}: count={data['count']}, avg_length={data['avg_length']:.1f}")
