"""Summarise a filing / annual report / results press release (txt, md or pdf) with a local model.

Map-reduce: summarise each chunk, then combine. Every number in the final summary is checked against the source text.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import SETTINGS, Settings
from . import ollama

CHUNK_CHARS = 6000
MAX_CHUNKS = 14

MAP_PROMPT = """Extract from this excerpt of a company filing ONLY items that are explicitly stated: revenue/profit/margin figures,
debt or cash changes, guidance, new risks, litigation, auditor qualifications, related-party transactions, management changes.
Copy numbers exactly as written. If the excerpt has none of these, reply exactly: nothing relevant. Do not add outside knowledge."""

REDUCE_PROMPT = """Combine these extracted notes into a concise summary (max 300 words) with headings: Financial highlights, Risks and red flags,
Management commentary, Open questions. Use only the notes; copy numbers exactly; if something is not in the notes say "not stated". No forecasts or advice."""


def read_text(path: Path | str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support needs: pip install pypdf   (or convert the file to .txt)") from exc
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    return p.read_text(errors="ignore")


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    paras, chunks, cur = re.split(r"\n\s*\n", text), [], ""
    for para in paras:
        if len(cur) + len(para) > size and cur:
            chunks.append(cur)
            cur = ""
        while len(para) > size:               # very long paragraph: hard split
            chunks.append(para[:size])
            para = para[size:]
        cur += para + "\n\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def _norm_numbers(text: str) -> set[str]:
    return {t.replace(",", "").rstrip("0").rstrip(".") if "." in t else t.replace(",", "")
            for t in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def unverified_numbers(summary: str, source: str) -> list[str]:
    src = _norm_numbers(source)
    bad = []
    for t in _norm_numbers(summary):
        v = float(t)
        if v.is_integer() and (v < 10 or 1990 <= v <= 2100):
            continue
        if t not in src:
            bad.append(t)
    return sorted(bad)


def summarize(path: Path | str, model: str | None = None, s: Settings = SETTINGS) -> dict:
    text = read_text(path)
    if len(text.strip()) < 200:
        raise ValueError("file has almost no extractable text (scanned PDF? try OCR first)")
    chunks = chunk_text(text)
    truncated = len(chunks) > MAX_CHUNKS
    notes = []
    for ch in chunks[:MAX_CHUNKS]:
        out = ollama.chat([{"role": "system", "content": MAP_PROMPT}, {"role": "user", "content": ch}], model, s=s)
        if "nothing relevant" not in out.lower():
            notes.append(out)
    if not notes:
        return {"summary": "No relevant financial content found in the processed text.", "unverified_numbers": [],
                "chunks": len(chunks), "truncated": truncated}
    summary = ollama.chat([{"role": "system", "content": REDUCE_PROMPT}, {"role": "user", "content": "\n\n".join(notes)}], model, s=s)
    return {"summary": summary, "unverified_numbers": unverified_numbers(summary, text), "chunks": len(chunks), "truncated": truncated}
