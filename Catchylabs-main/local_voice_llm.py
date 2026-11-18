"""Local Ollama adapter for voice-originated chat messages."""

from __future__ import annotations

import os

import httpx
from fastapi import HTTPException, status

from intelligence.language import output_language_instruction, resolve_language


VOICE_LOCAL_ONLY = os.environ.get("VOICE_LOCAL_ONLY", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
LOCAL_VOICE_LLM_MODEL = os.environ.get("LOCAL_VOICE_LLM_MODEL", "qwen3:4b").strip()
_OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
_OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


async def local_llm_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=0.7, trust_env=False) as client:
            response = await client.get(_OLLAMA_TAGS_URL)
        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        return any(
            isinstance(item, dict)
            and LOCAL_VOICE_LLM_MODEL in {item.get("name"), item.get("model")}
            for item in models
        )
    except (httpx.HTTPError, ValueError, TypeError):
        return False


async def generate_local_reply(
    messages: list[dict[str, str]],
    language_preference: str,
) -> dict[str, str]:
    latest_user_text = messages[-1]["content"]
    resolved_language = resolve_language(latest_user_text, language_preference)
    system_prompt = (
        "You are CatchyLab's local, provider-independent voice assistant. Answer clearly, "
        "preserve the user's language and intent, and do not claim to have performed research "
        "or actions that are absent from the conversation. "
        + output_language_instruction(resolved_language["language"])
    )
    payload = {
        "model": LOCAL_VOICE_LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "options": {"num_ctx": 8192, "num_predict": 1024},
    }
    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(_OLLAMA_CHAT_URL, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local language model timed out. Try a shorter message or a smaller local model.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local Ollama service is unavailable. Start Ollama; no cloud model fallback was used.",
        ) from exc

    if response.status_code == 404:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The configured local model '{LOCAL_VOICE_LLM_MODEL}' is not installed in Ollama. Install it explicitly; no cloud fallback was used.",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The local Ollama model returned an error. Check its local logs and retry.",
        )
    try:
        result = response.json()
        answer = result["message"]["content"].strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The local Ollama response was not in the expected chat format.",
        ) from exc
    if not answer:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The local Ollama model returned an empty response.",
        )
    return {"model": LOCAL_VOICE_LLM_MODEL, "text": answer}
