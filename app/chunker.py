from __future__ import annotations

from typing import Dict, List


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - self.overlap
        return chunks

    def create_chunks(self, parsed_docs: List[Dict]) -> List[Dict]:
        all_chunks: List[Dict] = []

        for doc in parsed_docs:
            chunks = self.chunk_text(doc["text"])
            for idx, chunk in enumerate(chunks, start=1):
                all_chunks.append(
                    {
                        "chunk_id": f'{doc["file_name"]}_p{doc.get("page", "na")}_c{idx}',
                        "file_name": doc["file_name"],
                        "file_path": doc["file_path"],
                        "page": doc.get("page"),
                        "section": doc.get("section"),
                        "doc_type": doc.get("doc_type"),
                        "text": chunk,
                    }
                )
        return all_chunks