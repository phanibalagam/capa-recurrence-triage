"""
Pluggable LLM backends.

Vendored into this project deliberately: each project in this collection is
self-contained and redistributable on its own, so nothing is imported across
project boundaries. If you fix a bug here, fix it in the sibling copy too.

Design rule for this whole repo: nothing is allowed to *require* a hosted LLM.
Every pipeline runs end-to-end with backend="extractive", which composes answers
only from retrieved source text. That keeps the projects runnable offline, and
it makes the citation-binding logic testable independently of any model.

Backends
--------
extractive : no model. Stitches retrieved sentences together. Always available.
anthropic  : Claude via the Messages API. Needs ANTHROPIC_API_KEY.
openai     : GPT via the Chat Completions API. Needs OPENAI_API_KEY.
ollama     : a local model served by Ollama at http://localhost:11434.

Usage
-----
    from src.llm import get_backend
    llm = get_backend("extractive")
    print(llm.complete(system="...", prompt="...", context_blocks=[...]))
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List


class LLMUnavailable(RuntimeError):
    """Raised when a backend is selected but cannot be reached."""


@dataclass
class ContextBlock:
    """One retrieved passage, with the identifier a citation must point at."""

    source_id: str
    text: str
    meta: dict = field(default_factory=dict)


class Backend:
    name = "base"

    def complete(self, system: str, prompt: str,
                 context_blocks: List[ContextBlock] | None = None,
                 max_tokens: int = 900) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------
# extractive
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class ExtractiveBackend(Backend):
    """
    Composes an answer purely by selecting sentences from the retrieved blocks.

    It cannot hallucinate, because it never generates a token that was not in a
    source document. That makes it a useful floor: any generative backend you
    plug in has to beat this on usefulness while matching it on faithfulness.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 8):
        self.max_sentences = max_sentences

    def complete(self, system: str, prompt: str,
                 context_blocks: List[ContextBlock] | None = None,
                 max_tokens: int = 900) -> str:
        blocks = context_blocks or []
        if not blocks:
            return ("INSUFFICIENT EVIDENCE - no source passages were retrieved "
                    "for this question.")

        query_terms = {w for w in re.findall(r"[a-z0-9]{4,}", prompt.lower())}

        scored = []
        for b in blocks:
            for sent in _SENT_SPLIT.split(b.text.strip()):
                sent = sent.strip()
                if len(sent) < 40:
                    continue
                terms = set(re.findall(r"[a-z0-9]{4,}", sent.lower()))
                overlap = len(query_terms & terms)
                if overlap == 0:
                    continue
                # normalise so long sentences do not win purely on length
                score = overlap / (len(terms) ** 0.5)
                scored.append((score, sent, b.source_id))

        if not scored:
            return ("INSUFFICIENT EVIDENCE - retrieved passages do not address "
                    "the question.")

        scored.sort(key=lambda t: -t[0])
        seen, lines = set(), []
        for _, sent, sid in scored:
            key = sent[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {sent} [{sid}]")
            if len(lines) >= self.max_sentences:
                break
        return "\n".join(lines)


# --------------------------------------------------------------------------
# hosted / local generative backends
# --------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMUnavailable(f"{url} returned {e.code}: {e.read()[:400]!r}") from e
    except Exception as e:  # noqa: BLE001 - surfaced to the caller as-is
        raise LLMUnavailable(f"could not reach {url}: {e}") from e


def _render_context(blocks: List[ContextBlock] | None) -> str:
    if not blocks:
        return "(no sources retrieved)"
    return "\n\n".join(f"[{b.source_id}]\n{b.text}" for b in blocks)


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

    def complete(self, system, prompt, context_blocks=None, max_tokens=900):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content":
                          f"SOURCES\n{_render_context(context_blocks)}\n\nTASK\n{prompt}"}],
        }
        data = _post_json("https://api.anthropic.com/v1/messages", body,
                          {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return "".join(p.get("text", "") for p in data.get("content", []))


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY", "")
        if not self.key:
            raise LLMUnavailable("OPENAI_API_KEY is not set")

    def complete(self, system, prompt, context_blocks=None, max_tokens=900):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content":
                 f"SOURCES\n{_render_context(context_blocks)}\n\nTASK\n{prompt}"},
            ],
        }
        data = _post_json("https://api.openai.com/v1/chat/completions", body,
                          {"Authorization": f"Bearer {self.key}"})
        return data["choices"][0]["message"]["content"]


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system, prompt, context_blocks=None, max_tokens=900):
        body = {
            "model": self.model,
            "system": system,
            "prompt": f"SOURCES\n{_render_context(context_blocks)}\n\nTASK\n{prompt}",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        return _post_json(f"{self.host}/api/generate", body, {})["response"]


_REGISTRY = {
    "extractive": ExtractiveBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def get_backend(name: str = "extractive", **kwargs) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_backends() -> dict:
    """Which backends can actually run right now, for the notebook to show."""
    status = {"extractive": "ready (no key needed)"}
    status["anthropic"] = "ready" if os.environ.get("ANTHROPIC_API_KEY") else "no ANTHROPIC_API_KEY"
    status["openai"] = "ready" if os.environ.get("OPENAI_API_KEY") else "no OPENAI_API_KEY"
    try:
        urllib.request.urlopen(
            os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=2)
        status["ollama"] = "ready"
    except Exception:  # noqa: BLE001
        status["ollama"] = "not running on localhost:11434"
    return status
