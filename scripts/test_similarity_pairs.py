import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import MockEmbedder
from src.chunking import compute_similarity
from bench import TfidfEmbedder, parse_markdown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pairs = [
    (
        "Quy định thời hạn mượn tài liệu về nhà của sinh viên",
        "Thời gian người học được phép giữ giáo trình mượn tại thư viện",
        "Đồng nghĩa, khác từ"
    ),
    (
        "Sinh viên được phép mang ba lô túi xách vào phòng đọc",
        "Sinh viên không được phép mang ba lô túi xách vào phòng đọc",
        "Trùng từ khóa, ngược nghĩa (phủ định)"
    ),
    (
        "Thủ tục gia hạn sách trực tuyến trên cổng thông tin",
        "Cách mượn giáo trình và trả tài liệu đúng hạn",
        "Cùng chủ đề thư viện, liên quan"
    ),
    (
        "Sinh viên tra cứu luận văn tại phòng đọc chuyên ngành",
        "Công thức nấu phở bò gia truyền Hà Nội thơm ngon",
        "Hoàn toàn khác chủ đề"
    ),
    (
        "Nội quy mượn trả tài liệu phòng mượn 102",
        "Nội quy mượn trả tài liệu phòng mượn 102",
        "Trùng khớp 100%"
    ),
]

# Fit TF-IDF on corpus
files = sorted(Path("data/thu-vien").glob("*.md"))
corpus_texts = []
for fp in files:
    _, body = parse_markdown(fp)
    corpus_texts.append(body)

tfidf = TfidfEmbedder()
tfidf.fit(corpus_texts)
mock = MockEmbedder()

print(f"{'Cặp':<4} | {'Mô tả':<35} | {'MockEmbedder':<12} | {'TfidfEmbedder':<12}")
print("-" * 75)
for i, (s1, s2, desc) in enumerate(pairs, 1):
    mock_sim = compute_similarity(mock(s1), mock(s2))
    tfidf_sim = compute_similarity(tfidf.encode(s1), tfidf.encode(s2))
    print(f"{i:<4} | {desc:<35} | {mock_sim:12.4f} | {tfidf_sim:12.4f}")
