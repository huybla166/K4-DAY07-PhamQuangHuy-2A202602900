from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Kho tri thức hiện đang trống. Không tìm thấy thông tin phù hợp."

        retrieved_docs = self.store.search(question, top_k=top_k)
        if not retrieved_docs:
            return "Không tìm thấy thông tin phù hợp trong kho tri thức."

        context_blocks: list[str] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("metadata", {}).get("doc_id", doc.get("id", f"doc_{i}"))
            content = doc.get("content", "").strip()
            context_blocks.append(f"[{i}] (Nguồn: {source})\n{content}")

        context_text = "\n\n".join(context_blocks)

        prompt = (
            "Bạn là trợ lý hỏi đáp dựa trên tri thức được cung cấp.\n\n"
            "NGỮ CẢNH:\n"
            f"{context_text}\n\n"
            "YÊU CẦU:\n"
            "- Chỉ sử dụng thông tin trong ngữ cảnh trên để trả lời câu hỏi.\n"
            "- Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy thông tin.\n"
            "- Trích dẫn nguồn thông tin tương ứng dạng [1], [2] khi trả lời.\n\n"
            f"CÂU HỎI: {question}\n\n"
            "CÂU TRẢ LỜI:"
        )

        return self.llm_fn(prompt)
