from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    data_dir: str = os.getenv("DATA_DIR", "data/raw_docs")
    processed_dir: str = os.getenv("PROCESSED_DIR", "data/processed")
    vector_dir: str = os.getenv("VECTOR_DIR", "vector_store")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))


settings = Settings()