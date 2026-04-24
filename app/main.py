from __future__ import annotations

import time

from app.answer_generator import AnswerGenerator
from app.config import settings
from app.embedder import Embedder
from app.language_utils import detect_language
from app.retriever import VectorRetriever


class KnowledgeAssistant:
    def __init__(self):
        self.embedder = Embedder(settings.embedding_model)
        self.retriever = VectorRetriever(settings.vector_dir)
        self.retriever.load()

        self.answer_generator = AnswerGenerator(
            ollama_url=settings.ollama_url,
            model_name=settings.ollama_model,
        )

    @staticmethod
    def score_to_percentage(score: float) -> float:
        percent = max(0.0, min(score * 100, 100.0))
        return round(percent, 2)

    def ask(self, query: str, explicit_language: str | None = None) -> dict:
        start_time = time.time()

        language = detect_language(query, explicit_language=explicit_language)

        query_embedding = self.embedder.encode_query(query)
        chunks = self.retriever.search(query_embedding, top_k=settings.top_k)

        retrieval_time = time.time() - start_time

        if not chunks:
            return {
                "language": language,
                "answer": "Mujhe is folder ke documents me iska exact answer nahi mila",
                "sources": [],
                "retrieved_chunks": [],
                "best_relevance_percent": 0,
                "avg_relevance_percent": 0,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(time.time() - start_time, 2),
            }

        best_score = chunks[0].get("score", 0)
        scores = [chunk.get("score", 0) for chunk in chunks]

        best_percent = self.score_to_percentage(best_score)
        avg_percent = round(sum(self.score_to_percentage(s) for s in scores) / len(scores), 2)

        if best_score < 0.25:
            return {
                "language": language,
                "answer": "Mujhe is folder ke documents me iska exact answer nahi mila",
                "sources": [],
                "retrieved_chunks": chunks,
                "best_relevance_percent": best_percent,
                "avg_relevance_percent": avg_percent,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(time.time() - start_time, 2),
            }

        generation_start = time.time()
        answer = self.answer_generator.generate(query, chunks, language)
        generation_time = time.time() - generation_start

        sources = []
        for chunk in chunks:
            if chunk.get("page") is not None:
                sources.append(f"{chunk['file_name']}, Page {chunk['page']}")
            else:
                sources.append(
                    f"{chunk['file_name']}, Section {chunk.get('section', 'N/A')}"
                )

        total_time = time.time() - start_time

        return {
            "language": language,
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),
            "retrieved_chunks": chunks,
            "best_relevance_percent": best_percent,
            "avg_relevance_percent": avg_percent,
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "total_time": round(total_time, 2),
        }


def main() -> None:
    assistant = KnowledgeAssistant()
    print("Personal Knowledge AI ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        result = assistant.ask(query)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nStats:")
        print(f"Best relevance: {result['best_relevance_percent']}%")
        print(f"Average relevance: {result['avg_relevance_percent']}%")
        print(f"Total time: {result['total_time']} seconds")

        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source}")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()