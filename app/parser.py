from __future__ import annotations

import os
from typing import Dict, List

import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)

    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    return cleaned.strip()


def parse_pdf(file_path: str) -> List[Dict]:
    pages: List[Dict] = []

    try:
        doc = fitz.open(file_path)

        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = clean_text(text)

            if text:
                pages.append(
                    {
                        "file_name": os.path.basename(file_path),
                        "file_path": file_path,
                        "page": i,
                        "section": f"Page {i}",
                        "text": text,
                        "doc_type": "pdf",
                    }
                )

        doc.close()

    except Exception as e:
        print(f"Skipping PDF due to error: {file_path} -> {e}")

    return pages


def parse_docx(file_path: str) -> List[Dict]:
    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = clean_text("\n".join(paragraphs))

        if not full_text:
            return []

        return [
            {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "page": None,
                "section": "Document Body",
                "text": full_text,
                "doc_type": "docx",
            }
        ]

    except Exception as e:
        print(f"Skipping DOCX due to error: {file_path} -> {e}")
        return []


def parse_txt(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = clean_text(f.read())

        if not text:
            return []

        return [
            {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "page": None,
                "section": "Text File",
                "text": text,
                "doc_type": "txt",
            }
        ]

    except Exception as e:
        print(f"Skipping TXT due to error: {file_path} -> {e}")
        return []


def parse_file(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    if ext == ".docx":
        return parse_docx(file_path)
    if ext == ".txt":
        return parse_txt(file_path)

    return []


def discover_files(folder_path: str) -> List[str]:
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    paths: List[str] = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, file_name))

    return sorted(paths)