from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

from app.chunker import TextChunker
from app.config import settings
from app.embedder import Embedder
from app.parser import discover_files, parse_file
from app.retriever import VectorRetriever


class DocumentIngestor:
    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedder = Embedder(settings.embedding_model)
        self.retriever = VectorRetriever(settings.vector_dir)

    def index_exists(self) -> bool:
        index_path = os.path.join(settings.vector_dir, "faiss.index")
        meta_path = os.path.join(settings.vector_dir, "chunks_metadata.json")
        return os.path.exists(index_path) and os.path.exists(meta_path)

    def clear_old_index(self) -> None:
        if os.path.exists(settings.vector_dir):
            shutil.rmtree(settings.vector_dir)

        if os.path.exists(settings.processed_dir):
            shutil.rmtree(settings.processed_dir)

        os.makedirs(settings.vector_dir, exist_ok=True)
        os.makedirs(settings.processed_dir, exist_ok=True)

    def parse_all_documents(self, folder_path: str) -> List[Dict]:
        file_paths = discover_files(folder_path)
        all_parsed_docs: List[Dict] = []

        print(f"Found {len(file_paths)} supported files.")

        for path in file_paths:
            print(f"Reading: {path}")
            parsed = parse_file(path)

            if parsed:
                all_parsed_docs.extend(parsed)
            else:
                print(f"Skipped or empty: {path}")

        return all_parsed_docs

    def save_processed_docs(self, parsed_docs: List[Dict], chunks: List[Dict]) -> None:
        os.makedirs(settings.processed_dir, exist_ok=True)

        parsed_path = os.path.join(settings.processed_dir, "parsed_docs.json")
        chunks_path = os.path.join(settings.processed_dir, "chunks.json")

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(parsed_docs, f, ensure_ascii=False, indent=2)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def run(self, folder_path: str | None = None, force: bool = False) -> Dict:
        folder = os.path.abspath(folder_path or settings.data_dir)

        if self.index_exists() and not force:
            return {
                "status": "skipped",
                "message": "Index already exists. Skipped ingestion.",
            }

        if force:
            self.clear_old_index()

        parsed_docs = self.parse_all_documents(folder)

        if not parsed_docs:
            raise ValueError("No readable text found in supported documents.")

        chunks = self.chunker.create_chunks(parsed_docs)

        if not chunks:
            raise ValueError("No chunks created from documents.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode_texts(texts)

        self.save_processed_docs(parsed_docs, chunks)
        self.retriever.save(embeddings, chunks)

        return {
            "status": "success",
            "message": "Ingestion completed successfully.",
            "parsed_units": len(parsed_docs),
            "chunks": len(chunks),
            "folder": folder,
        }


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    result = ingestor.run(force=True)
    print(result)