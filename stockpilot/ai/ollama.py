"""Thin client for a local Ollama server (no data leaves your machine)."""
from __future__ import annotations

from ..config import SETTINGS, Settings


class OllamaError(RuntimeError):
    pass


def is_available(s: Settings = SETTINGS) -> bool:
    import requests
    try:
        return requests.get(f"{s.ollama_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2, s: Settings = SETTINGS) -> str:
    import requests
    model = model or s.ollama_model
    try:
        r = requests.post(f"{s.ollama_url}/api/chat", timeout=s.ollama_timeout,
                          json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}})
    except requests.RequestException as exc:
        raise OllamaError(f"Cannot reach Ollama at {s.ollama_url} ({exc}). Is `ollama serve` running?") from exc
    if r.status_code == 404:
        raise OllamaError(f"Model '{model}' not found. Run: ollama pull {model}   (or set STOCKPILOT_OLLAMA_MODEL)")
    if not r.ok:
        raise OllamaError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()["message"]["content"].strip()
    except (KeyError, ValueError) as exc:
        raise OllamaError("Unexpected response format from Ollama") from exc
