"""Load documents and split them into overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return a list of (text, page_number) pairs for a file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported file type: {suffix}")


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size character pieces, preferring paragraph/sentence
    boundaries, with `overlap` characters carried between neighbouring chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = clean(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to end on a natural boundary in the back half of the window.
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_file(path: Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for text, page in load_file(path):
        for i, piece in enumerate(split_text(text, chunk_size, overlap)):
            cid = f"{path.name}:{page or 0}:{i}"
            out.append(Chunk(text=piece, source=path.name, page=page, chunk_id=cid))
    return out


def chunk_directory(directory: Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            chunks.extend(chunk_file(path, chunk_size, overlap))
    return chunks
