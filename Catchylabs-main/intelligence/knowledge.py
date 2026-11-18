"""Versioned, provider-independent retrieval for CatchyLab's original playbooks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


KNOWLEDGE_VERSION = "1.1.0"
KNOWLEDGE_FILE = Path(__file__).with_name("knowledge_base") / "v1" / "marketing.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0900-\u097f]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "where", "which", "with", "you",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in _STOP_WORDS
    }


def _load_documents() -> list[dict[str, Any]]:
    with KNOWLEDGE_FILE.open(encoding="utf-8") as file:
        content = json.load(file)
    if content.get("version") != KNOWLEDGE_VERSION:
        raise ValueError("Marketing knowledge file version does not match the loader.")
    documents = content.get("documents")
    if not isinstance(documents, list) or not all(isinstance(item, dict) for item in documents):
        raise ValueError("Marketing knowledge file must contain a documents list.")
    corpus_metadata = content.get("corpus_metadata", {})
    if not isinstance(corpus_metadata, dict):
        raise ValueError("Marketing knowledge corpus metadata must be an object.")
    return [
        {
            **document,
            "metadata": {
                **corpus_metadata,
                **(document.get("metadata") if isinstance(document.get("metadata"), dict) else {}),
            },
        }
        for document in documents
    ]


def retrieve_knowledge(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return original guidance ranked by token coverage; no external model is called."""
    if not isinstance(query, str) or not query.strip() or limit < 1:
        return []

    terms = _tokens(query)
    if not terms:
        return []

    ranked: list[tuple[float, dict[str, Any]]] = []
    for document in _load_documents():
        searchable = " ".join(
            [
                str(document.get("title", "")),
                str(document.get("content", "")),
                " ".join(str(item) for item in document.get("keywords", [])),
            ]
        )
        document_terms = _tokens(searchable)
        overlap = terms & document_terms
        if not overlap:
            continue

        # Coverage rewards documents that address more of the request; the
        # bounded frequency bonus favors repeated, salient framework terms.
        score = len(overlap) / len(terms)
        score += sum(searchable.lower().count(term) for term in overlap) / max(
            len(document_terms) * 10, 1
        )
        ranked.append((score, document))

    ranked.sort(key=lambda result: (-result[0], result[1].get("id", "")))
    return [
        {
            "id": document["id"],
            "title": document["title"],
            "domain": document["domain"],
            "version": KNOWLEDGE_VERSION,
            "content": document["content"],
            "metadata": document["metadata"],
            "source": {
                "name": "CatchyLab original working synthesis",
                "type": "internal_playbook",
                "attribution": "CatchyLab",
            },
            "relevance_score": round(score, 4),
        }
        for score, document in ranked[:limit]
    ]


def append_knowledge_guidance(prompt: str, query: str, limit: int = 2) -> str:
    """Append selected internal playbook guidance without presenting it as evidence."""
    documents = retrieve_knowledge(query, limit=limit)
    if not documents:
        return prompt

    guidance = "\n\n".join(
        f"[{document['domain']}] {document['title']}\n{document['content']}"
        for document in documents
    )
    return (
        f"{prompt}\n\n"
        "CATCHYLAB INTERNAL FRAMEWORK GUIDANCE (not external evidence; "
        "do not cite as a source):\n"
        f"{guidance}"
    )
