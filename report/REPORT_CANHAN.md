# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Quang Huy
**Nhóm:** Mãi là Lốp Trưởng
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:*
> Độ tương tự cosine cao nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ (giá trị cosine tiến gần về 1), thể hiện rằng hai đoạn văn bản có sự tương đồng sâu sắc về mặt ngữ nghĩa và hướng biểu đạt chủ đề.

**Ví dụ có độ tương tự CAO:**
- Câu A: Cậu bé dắt chú cún cưng đi dạo trong hoa viên.
- Câu B: Đứa trẻ dẫn con chó đi bách bộ ngoài công viên.
- Tại sao tương đồng: Dù hai câu sử dụng bộ từ vựng hoàn toàn khác nhau (cậu bé / đứa trẻ, cún cưng / con chó, dắt đi dạo / dẫn đi bách bộ, hoa viên / công viên), mô hình embedding vẫn nhận ra chúng cùng diễn tả một hành động và ý nghĩa tương đương, thể hiện khả năng hiểu nghĩa ngữ cảnh thay vì so khớp từ vựng đơn thuần.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Thuật toán lan truyền ngược dùng để cập nhật trọng số trong mạng nơ-ron sâu.
- Câu B: Bí quyết nướng bánh mì phô mai giòn xốp bằng lò nướng nhiệt độ cao.
- Tại sao khác: Hai câu thuộc hai lĩnh vực tri thức hoàn toàn biệt lập (trí tuệ nhân tạo và ẩm thực làm bánh), dẫn đến vector biểu diễn ngữ nghĩa hướng về các vùng không gian trực giao nhau (cosine gần bằng 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:*
> Khoảng cách Euclid phụ thuộc vào độ lớn (magnitude/norm) của vector — vốn dễ bị lệch do độ dài văn bản hoặc số lượng từ tích lũy. Ngược lại, cosine similarity chỉ tập trung đo góc (hướng) biểu diễn ngữ nghĩa, phản ánh đúng chủ đề và nội dung độc lập với độ dài văn bản, đồng thời được chuẩn hóa tự nhiên trên thang [-1, 1].

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy (step) = chunk_size - overlap = 500 - 50 = 450 ký tự.
> - Số lượng chunk = ceil((độ_dài_tài_liệu - overlap) / (chunk_size - overlap)) = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.111...) = 23.
> - Kiểm tra lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` cho kết quả chính xác 23 chunks (các vị trí bắt đầu: 0, 450, ..., 9900).
> *Đáp án:* 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:*
> - Khi overlap tăng lên 100: bước nhảy step = 500 - 100 = 400 ký tự; số chunk = ceil((10,000 - 100) / 400) = ceil(9,900 / 400) = ceil(24.75) = 25 chunks (tăng thêm 2 chunks).
> - Người ta muốn overlap lớn hơn nhằm bảo toàn trọn vẹn ngữ cảnh tại các ranh giới cắt, tránh để một câu, mệnh đề hoặc định nghĩa quan trọng bị xẻ đôi giữa hai chunk liền kề, qua đó cải thiện chất lượng truy xuất (retrieval recall) khi câu hỏi truy vấn rơi vào vùng giáp ranh.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*
> Sử dụng regex Positive Lookbehind `r"(?<=[.!?])\s+"` để tách câu ngay sau dấu kết thúc kèm khoảng trắng, giúp bảo toàn toàn bộ dấu câu không bị mất như regex split thông thường; sau đó gom tối đa `max_sentences_per_chunk` câu thành một chunk. Đã xử lý ngoại lệ text rỗng/khoảng trắng trả về `[]`, strip khoảng trắng thừa. Edge case chưa xử lý được: các từ viết tắt có dấu chấm (`TS.`, `ThS.`, `v.v.`) hoặc số thập phân (`3.14`) sẽ bị nhận nhầm thành ranh giới kết câu.

**`RecursiveChunker.chunk` / `_split`** — Hướng tiếp cận & Kiến trúc chi tiết:

#### 1. Tư duy thiết kế & Mục tiêu kỹ thuật
Trong các hệ thống RAG (Retrieval-Augmented Generation), bài toán chia nhỏ văn bản luôn phải đánh đổi giữa **tính toàn vẹn ngữ nghĩa (semantic coherence)** và **giới hạn kích thước ngữ cảnh (token/character budget)**:
- `FixedSizeChunker`: Cắt mù quáng theo số lượng ký tự cố định, dễ gây đứt gãy giữa từ hoặc xẻ đôi câu/đoạn văn quan trọng.
- `SentenceChunker`: Giữ được câu, nhưng khi gặp câu văn quá dài hoặc đoạn mô tả kỹ thuật phức tạp sẽ làm tràn kích thước chunk mong muốn.
- **`RecursiveChunker`** ra đời nhằm dung hòa cả hai: Ưu tiên tối đa việc giữ nguyên các khối ngữ nghĩa lớn nhất có thể (đoạn văn $\to$ dòng $\to$ câu $\to$ từ), và chỉ thực hiện phân tách sâu hơn khi khối hiện tại thực sự vượt quá ngưỡng `chunk_size`.

#### 2. Nguyên lý thuật toán 2 chiều (Bidirectional Mechanism)
Thuật toán không dừng lại ở việc cắt nhỏ (Top-down) mà bắt buộc phải có chiều gom lên (Bottom-up) để đảm bảo mật độ thông tin:

- **Chiều 1: Phân tách đệ quy từ trên xuống (Top-down Recursive Splitting):**
  - Hệ thống duy trì danh sách separator có thứ tự ưu tiên giảm dần: `["\n\n", "\n", ". ", " ", ""]`.
  - Văn bản được tách trước hết bằng separator ở cấp độ cao nhất (`sep = remaining_separators[0]`).
  - Với mỗi mảnh con thu được:
    - Nếu độ dài mảnh $\le \text{chunk\_size}$: Giữ nguyên mảnh đó.
    - Nếu độ dài mảnh $> \text{chunk\_size}$: Đệ quy gọi tiếp `_split(piece, remaining_separators[1:])` để hạ cấp xuống separator nhỏ hơn.
    - Nếu separator hiện tại không xuất hiện trong văn bản: Tự động chuyển tiếp ngay sang separator kế tiếp để tránh thao tác tách dư thừa.

- **Chiều 2: Gom cụm từ dưới lên (Bottom-up Greedy Merging):**
  - Đây là khâu then chốt: Sau khi các nhánh đệ quy trả về các mảnh con nhỏ hơn `chunk_size`, nếu không gom lại thì văn bản nhiều dòng ngắn (như danh sách, code, thơ) sẽ bị xé thành hàng trăm chunk vụn 5–10 ký tự, làm suy thoái nghiêm trọng điểm tương đồng khi truy xuất vector.
  - Bộ gom (Greedy Aggregator) duyệt tuần tự qua các mảnh con, tích lũy chúng vào `current_chunk` kèm separator tương ứng cho đến khi việc thêm mảnh tiếp theo sẽ vượt quá `chunk_size`. Khi đó, `current_chunk` được chốt lại và mở một chunk mới.

#### 3. Phân tích chi tiết 3 Base Cases (Trường hợp cơ sở) & Edge Cases
Để thuật toán chạy ổn định và không rơi vào đệ quy vô hạn:
1. **Base Case 1 (`not current_text`):** Văn bản rỗng hoặc đã duyệt hết $\to$ lập tức trả về danh sách rỗng `[]`.
2. **Base Case 2 (`len(current_text) <= chunk_size`):** Đoạn văn bản hiện tại đã nằm gọn trong giới hạn kích thước $\to$ không cần phân tách thêm, trả về `[current_text]` để giữ trọn ngữ cảnh.
3. **Base Case 3 (`not remaining_separators` hoặc `sep == ""`):** Khi đã thử hết toàn bộ danh sách separator (kể cả trường hợp người dùng truyền trực tiếp `separators=[]`) mà độ dài chuỗi vẫn lớn hơn `chunk_size` $\to$ kích hoạt cơ chế dự phòng (Fallback Slicing): Cắt lát cưỡng bức theo từng đoạn có độ dài `chunk_size` (`[text[i:i+chunk_size]...]`).

#### 4. Sơ đồ kiến trúc & Luồng xử lý dữ liệu (Architectural Flow)

```
                            [ Raw Text Input ]
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   Entrypoint: chunk(text)    │
                     │  - Kiểm tra rỗng & tiền xử lý│
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │    Base Case Evaluator       │
                     │  1. text rỗng       -> []    │
                     │  2. len <= size     -> [text]│
                     │  3. hết separator   -> slice │
                     └──────────────┬───────────────┘
                                    │ (len > chunk_size & còn seps)
                                    ▼
       ═══════════════ CHIỀU 1: TOP-DOWN DECOMPOSITION ═══════════════
                     Tách theo separator ưu tiên Sep[0]:
          ["\n\n" (Đoạn)] → ["\n" (Dòng)] → [". " (Câu)] → [" " (Từ)]
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
              [Mảnh <= chunk_size]    [Mảnh > chunk_size]
                        │                       │ (Đệ quy với Next Seps)
                        │             ┌─────────▼─────────┐
                        │             │ _split(sub_piece, │
                        │             │     next_seps)    │
                        │             └─────────┬─────────┘
                        └───────────┬───────────┘
                                    │
       ════════════════ CHIỀU 2: BOTTOM-UP MERGING ════════════════
                                    │
                     ┌──────────────▼───────────────┐
                     │   Greedy Chunk Aggregator    │
                     │  Nối tuần tự: piece + sep    │
                     │  sao cho tổng len tiệm cận   │
                     │  chunk_size tối ưu           │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                      [ Final List of Text Chunks ]
```

#### 5. Đánh giá chất lượng trong thực tế
- **Độ phức tạp thời gian (Time Complexity):** Trung bình đạt $\mathcal{O}(N)$ với $N$ là độ dài ký tự của văn bản do mỗi ký tự chỉ qua bước split và merge một số hằng số lần tương ứng với số tầng separator (tối đa 5 tầng).
- **Khả năng ứng dụng:** Cực kỳ hiệu quả cho tài liệu có cấu trúc phân tầng như quy chế, hợp đồng pháp lý, tài liệu kỹ thuật Markdown, và bài viết học thuật. Đảm bảo chunk sinh ra vừa vặn với kích thước context window của LLM mà không làm đứt đoạn logic ngữ nghĩa.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*
> Sử dụng cấu trúc thuần in-memory bằng danh sách các bản ghi từ điển chuẩn hóa (`_make_record` gồm `id`, `content`, `metadata` được copy an toàn, và `embedding`). Do vector embedding đã được chuẩn hóa L2, hàm `_search_records` tính độ tương tự cosine nhanh chóng bằng tích vô hướng (dot product) qua hàm `_dot`, sắp xếp giảm dần theo điểm số và trả về `top_k` kết quả (đã loại bỏ trường embedding để output gọn sạch).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*
> Bắt buộc phải lọc TRƯỚC (pre-filtering) trên toàn bộ kho lưu trữ để chọn ra tập ứng viên thỏa mãn tất cả các tiêu chí trong `metadata_filter`, sau đó mới thực hiện tìm kiếm tương tự (nếu lọc sau top-k, $k$ vị trí có thể bị chiếm hết bởi tài liệu sai dẫn đến trả về 0 kết quả dù tài liệu đúng vẫn tồn tại). Hàm `delete_document` lọc bỏ mọi bản ghi có `metadata['doc_id'] == doc_id` hoặc `id == doc_id`, trả về `True` nếu kích thước store giảm đi và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*
> Thực hiện theo quy trình 3 nhịp: (1) Kiểm tra store rỗng thì thông báo ngay tránh gọi LLM vô ích, (2) Tìm kiếm top-k chunk liên quan nhất, và (3) Dựng prompt đưa ngữ cảnh vào có đánh số `[1]`, `[2]` kèm nguồn `(Nguồn: doc_id)` nhằm đáp ứng tiêu chí Truy vết nguồn gốc (Source Traceability). Bổ sung ràng buộc khắt khe chống hallucination (bịa đặt thông tin): chỉ trả lời dựa trên ngữ cảnh được cung cấp, nêu rõ nếu không tìm thấy và yêu cầu trích dẫn số thứ tự nguồn tương ứng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CODE\AITHUCCHIEN\LABS\K4-DAY07-PhamQuangHuy-2A202602900
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42


---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (TF-IDF / Mock) | Đúng? |
|------|-----------|-----------|---------|------------------------------|-------|
| 1 | Quy định thời hạn mượn tài liệu về nhà của sinh viên | Thời gian người học được phép giữ giáo trình mượn tại thư viện | Cao (đồng nghĩa) | 0.1056 / -0.1172 | Sai (rất thấp do khác từ vựng) |
| 2 | Sinh viên được phép mang ba lô túi xách vào phòng đọc | Sinh viên không được phép mang ba lô túi xách vào phòng đọc | Thấp (ngược nghĩa/phủ định) | 0.9710 / -0.0412 | Sai (rất cao do trùng hầu hết từ vựng) |
| 3 | Thủ tục gia hạn sách trực tuyến trên cổng thông tin | Cách mượn giáo trình và trả tài liệu đúng hạn | Trung bình / Cao (cùng chủ đề thư viện) | 0.0599 / -0.1181 | Sai (thấp vì ít từ khóa chung) |
| 4 | Sinh viên tra cứu luận văn tại phòng đọc chuyên ngành | Công thức nấu phở bò gia truyền Hà Nội thơm ngon | Thấp (hoàn toàn khác chủ đề) | 0.0000 / -0.1884 | Đúng (hoàn toàn trực giao 0.0) |
| 5 | Nội quy mượn trả tài liệu phòng mượn 102 | Nội quy mượn trả tài liệu phòng mượn 102 | Rất cao (trùng khớp 100%) | 1.0000 / 1.0000 | Đúng (tuyệt đối 1.0) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là **Cặp 2**: hai câu có ý nghĩa logic hoàn toàn đối nghịch nhau (*được phép* vs *không được phép*) nhưng độ tương tự TF-IDF thực tế lại đạt tới **0.9710** (gần như tương đồng tuyệt đối), trong khi **Cặp 1** diễn đạt cùng một ý nhưng dùng từ đồng nghĩa khác nhau thì điểm chỉ đạt **0.1056**.  
> Điều này phơi bày hạn chế cố hữu của các mô hình dựa trên tần suất từ (TF-IDF/BM25) và cơ chế băm MockEmbedder: chúng chỉ đo sự trùng lặp bề mặt ký tự (lexical overlap), hoàn toàn mù quáng trước ngữ nghĩa sâu (semantics) và cấu trúc phủ định logic. Để hệ thống RAG hoạt động tin cậy, bắt buộc phải sử dụng các mô hình Dense Semantic Embedding thực thụ (như Sentence Transformers).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

*Chiến lược áp dụng: **RecursiveChunker** (chunk_size=300, separators=['\n\n', '\n', '. ', ' ']). Dữ liệu: 10 file markdown thư viện (62 chunks).*

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn mượn sách đối với sinh viên trong trường là bao nhiêu ngày và được gia hạn mấy lần? (filter: `audience=student`) | `library-borrow-home#1`: Chính sách mượn trả chung: tối đa 08 cuốn, thời hạn 60 ngày, gia hạn 1 lần 30 ngày... | 0.587 | Chưa (Top-3 thiếu gold `huit-borrow-student`, rơi vào failure case do trùng từ vựng mượn sách) | Agent tổng hợp nhầm theo quy định chung (60 ngày/08 cuốn) thay vì quy định sinh viên (10 ngày/03 cuốn). |
| 2 | Sinh viên được mượn tối đa bao nhiêu cuốn giáo trình tại Phòng 111 và thời hạn mượn là bao lâu? | `library-room-111#0`: Quy trình mượn trả tại Phòng 111, giáo trình mượn tối đa 8 cuốn, thời hạn 90 ngày, gia hạn 1 lần 30 ngày. | 0.463 | Có (Top 1) | Chính xác: Mỗi bạn đọc được mượn tối đa 8 cuốn giáo trình tại Phòng 111 trong 90 ngày, gia hạn 1 lần (30 ngày), tổng tối đa 120 ngày. |
| 3 | Tại Phòng 102 mượn sách tham khảo, số lượng mượn tối đa là bao nhiêu và được gia hạn mấy ngày? | `huit-borrow-student#1`: Quy định người sử dụng ngoài trường (1 cuốn, 10 ngày). (Gold lọt vào Top 2 & Top 3: `library-borrow-home#1`, `library-room-102#1`). | 0.522 | Có (Top 2 & 3) | Đầy đủ: Sinh viên được mượn tối đa 5 cuốn sách tham khảo tại Phòng 102, thời hạn 30 ngày, được gia hạn 1 lần trong thời gian 7 ngày. |
| 4 | Mức bồi thường khi làm mất tài liệu thư viện không còn phát hành trên thị trường được tính thế nào? | `library-lost-document#1`: Mức bồi thường trên cơ sở thời giá thị trường, tài liệu không phát hành thị trường đền gấp 03 lần giá bìa... | 0.429 | Có (Top 1, 2, 3 trọn bộ) | Chính xác tuyệt đối: Bồi thường bằng tiền gấp 03 lần giá bìa; nếu tài liệu còn phát hành thì đền tài liệu mới tương đương kèm 20.000đ lệ phí xử lý kỹ thuật. |
| 5 | Quy định mượn tài liệu in-house đọc tại chỗ cho phép mượn tối đa mấy cuốn và trả trước mấy giờ? | `library-inhouse-borrow#0`: Quy định mượn in-house tại chỗ đọc trong ngày, tối đa 02 cuốn, hoàn trả trước 17h30 tại Phòng 411. | 0.522 | Có (Top 1) | Chính xác: Mượn tối đa 02 cuốn đọc tại chỗ và phải trả lại quầy thủ thư Phòng 411 trước 17h30 cùng ngày. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 câu (Q2, Q3, Q4, Q5). Tổng điểm benchmark đạt **7/10 điểm**.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua việc đối chiếu kết quả demo giữa 4 thành viên, tôi nhận thấy chiến lược `FixedSizeChunker` có overlap (Thành viên 1 - 9/10 điểm) xử lý rất mượt mà các câu trả lời nằm sát ranh giới phân tách nhờ cửa sổ trượt 50 ký tự; trong khi `HeadingChunker` (Thành viên 3) giữ lại tiêu đề cấp cha (`#`, `##`) cực kỳ lợi hại trong việc duy trì ngữ cảnh cho các văn bản quy định dạng Điều/Khoản.  
> Chiến lược `RecursiveChunker` của tôi tuy giữ trọn vẹn ngữ nghĩa khối văn bản tự nhiên nhưng khi kết hợp với TF-IDF (không có overlap) dễ bị mất điểm ở các câu hỏi có mật độ từ khóa chung dày đặc. Đây là bài học thực tế quý giá về việc kết hợp linh hoạt giữa chunking phân cấp và overlap cửa sổ trượt.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **58 / 60** |

