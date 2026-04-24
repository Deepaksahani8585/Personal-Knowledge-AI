from __future__ import annotations

import json
import os
from typing import Dict, List

import faiss
import numpy as np


class VectorRetriever:
    def __init__(self, vector_dir: str):
        self.vector_dir = vector_dir
        self.index_path = os.path.join(vector_dir, "faiss.index")
        self.meta_path = os.path.join(vector_dir, "chunks_metadata.json")
        self.index = None
        self.metadata: List[Dict] = []

    def save(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        os.makedirs(self.vector_dir, exist_ok=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = metadata

    def load(self) -> None:
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Vector index not found. Run ingestion first.")

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.load()

        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results