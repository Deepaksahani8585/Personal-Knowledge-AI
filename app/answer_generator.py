from __future__ import annotations

from typing import Dict, List
import requests


REFUSAL_PATTERNS = [
    "i can't answer",
    "i cannot answer",
    "i can’t help",
    "i can't help",
    "sorry",
    "i am unable",
    "i cannot provide",
]

EXACT_NOT_FOUND_LINE = "mujhe is folder ke documents me iska exact answer nahi mila"


class AnswerGenerator:
    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url
        self.model_name = model_name

    @staticmethod
    def detect_answer_style(user_query: str) -> str:
        q = user_query.lower().strip()

        detailed_keywords = [
            "explain", "detail", "detailed", "full", "step by step",
            "how", "why", "process", "compare", "difference",
            "advantages", "disadvantages", "example", "examples",
            "strategy", "workflow", "complete", "deeply", "in depth",
            "long", "elaborate", "samjhao", "kaise", "kyun",
            "poora", "pura",
        ]

        short_keywords = [
            "short", "brief", "one line", "definition", "meaning",
            "what is", "who is", "when", "kya hai",
        ]

        if any(word in q for word in detailed_keywords):
            return "detailed"

        if any(word in q for word in short_keywords):
            return "short"

        if len(q.split()) <= 6:
            return "short"

        return "balanced"

    @staticmethod
    def build_context(chunks: List[Dict]) -> str:
        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            file_name = chunk.get("file_name", "Unknown File")
            page = chunk.get("page")
            section = chunk.get("section", "N/A")

            source = f"File: {file_name}"

            if page is not None:
                source += f", Page: {page}"
            else:
                source += f", Section: {section}"

            context_parts.append(
                f"[SOURCE {i}]\n"
                f"{source}\n"
                f"TEXT:\n{chunk.get('text', '')}\n"
            )

        return "\n".join(context_parts)

    @staticmethod
    def build_prompt(user_query: str, context: str, language: str) -> str:
        answer_style = AnswerGenerator.detect_answer_style(user_query)

        return f"""
You are a STRICT document-based AI assistant.

ABSOLUTE RULES:
- Use ONLY the given document context.
- Do NOT use outside knowledge.
- Do NOT invent anything.
- Do NOT add external links, websites, GitHub links, or internet sources.
- Do NOT mention anything that is not written in the context.
- Do NOT use placeholder sources like "file.pdf".
- Use ONLY the actual source names and page numbers shown in the context.
- If answer is found, extract all relevant information from the context.
- If answer is not found, say exactly:
Mujhe is folder ke documents me iska exact answer nahi mila
- If answer is present, NEVER include the not-found sentence.

ANSWER STYLE: {answer_style}

If SHORT:
- Give 2-3 useful bullet points.
- Keep it simple but not incomplete.

If BALANCED:
- Give 4-6 useful bullet points.
- Include the main idea and supporting points.

If DETAILED:
- Use headings and bullets.
- Explain properly using all relevant context.
- Include summary if useful.

LANGUAGE RULE:
- english = English.
- hindi = Hindi.
- hinglish = simple Hinglish.
- Preserve meaning from the document.

FORMAT RULE:
- Use bullet points.
- Every important point should use document context only.
- Do not invent source names.
- Do not invent page numbers.

Target language:
{language}

User question:
{user_query}

Document context:
{context}
""".strip()

    @staticmethod
    def is_refusal(answer: str) -> bool:
        text = answer.lower()
        return any(pattern in text for pattern in REFUSAL_PATTERNS)

    @staticmethod
    def clean_mixed_not_found(answer: str) -> str:
        cleaned_lines = []

        for line in answer.splitlines():
            if EXACT_NOT_FOUND_LINE not in line.lower():
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def remove_invalid_lines(answer: str) -> str:
        cleaned = []

        banned_terms = [
            "http",
            "https",
            "github",
            "www.",
            ".com",
            "external source",
            "internet",
            "source: file.pdf",
            "file.pdf, page",
        ]

        for line in answer.splitlines():
            lower = line.lower()

            if any(term in lower for term in banned_terms):
                continue

            cleaned.append(line)

        return "\n".join(cleaned).strip()

    @staticmethod
    def attach_verified_sources(answer: str, chunks: List[Dict]) -> str:
        sources = []

        for chunk in chunks:
            file_name = chunk.get("file_name", "Unknown File")
            page = chunk.get("page")
            section = chunk.get("section", "N/A")

            if page is not None:
                sources.append(f"{file_name}, Page {page}")
            else:
                sources.append(f"{file_name}, Section {section}")

        sources = list(dict.fromkeys(sources))

        if not sources:
            return answer

        source_block = "\n".join([f"- {source}" for source in sources])

        return f"{answer.strip()}\n\nVerified Sources:\n{source_block}"

    @staticmethod
    def fallback_answer(chunks: List[Dict], language: str) -> str:
        if language == "english":
            intro = "I found these relevant document excerpts:"
        elif language == "hindi":
            intro = "Documents में ये relevant हिस्से मिले:"
        else:
            intro = "Documents me ye relevant parts mile:"

        lines = [intro, ""]

        for chunk in chunks:
            file_name = chunk.get("file_name", "Unknown File")
            page = chunk.get("page")
            section = chunk.get("section", "N/A")
            text = chunk.get("text", "").strip()

            if page is not None:
                source = f"{file_name}, Page {page}"
            else:
                source = f"{file_name}, Section {section}"

            excerpt = text[:1200]

            lines.append(f"- Source: {source}")
            lines.append(f"  {excerpt}...")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def answer_too_short(answer: str) -> bool:
        cleaned = answer.strip()

        if len(cleaned) < 80:
            return True

        bullet_count = cleaned.count("- ") + cleaned.count("•")

        if bullet_count == 0 and len(cleaned.split()) < 25:
            return True

        return False

    def generate(self, user_query: str, chunks: List[Dict], language: str) -> str:
        context = self.build_context(chunks)
        prompt = self.build_prompt(user_query, context, language)

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.6,
                        "num_predict": 650,
                    },
                },
                timeout=240,
            )

            response.raise_for_status()
            answer = response.json().get("response", "").strip()

            if not answer:
                return self.fallback_answer(chunks, language)

            answer = self.remove_invalid_lines(answer)

            if not answer:
                return self.fallback_answer(chunks, language)

            if EXACT_NOT_FOUND_LINE in answer.lower():
                cleaned_answer = self.clean_mixed_not_found(answer)

                if cleaned_answer:
                    answer = cleaned_answer
                else:
                    return self.fallback_answer(chunks, language)

            if self.is_refusal(answer):
                return self.fallback_answer(chunks, language)

            if self.answer_too_short(answer):
                return self.fallback_answer(chunks, language)

            answer = self.attach_verified_sources(answer, chunks)
            return answer

        except Exception as exc:
            print(f"Ollama generation error: {exc}")
            return self.fallback_answer(chunks, language)