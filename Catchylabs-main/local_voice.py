"""HTTP routes that connect the replaceable local voice adapters."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

import local_speech_to_text
import local_voice_llm


router = APIRouter(prefix="/voice", tags=["Local voice"])
MAX_VOICE_MESSAGES = 20
MAX_VOICE_CONTEXT_CHARACTERS = 30_000
_AUDIO_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


class LocalVoiceMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class LocalVoiceChatRequest(BaseModel):
    messages: list[LocalVoiceMessage] = Field(min_length=1, max_length=MAX_VOICE_MESSAGES)
    language_preference: Literal["auto", "english", "hindi", "hinglish"] = "auto"

    @field_validator("messages")
    @classmethod
    def validate_context(cls, messages: list[LocalVoiceMessage]) -> list[LocalVoiceMessage]:
        if messages[-1].role != "user":
            raise ValueError("The final voice message must be from the user.")
        if sum(len(message.content) for message in messages) > MAX_VOICE_CONTEXT_CHARACTERS:
            raise ValueError("Voice conversation context is too long.")
        return messages


@router.get("/status")
async def get_local_voice_status() -> dict[str, Any]:
    """Report local runtime availability without downloading or initializing models."""
    transcription = local_speech_to_text.local_transcription_status()
    local_llm_ready = await local_voice_llm.local_llm_available()
    return {
        "local_only": local_voice_llm.VOICE_LOCAL_ONLY,
        "transcription_ready": transcription["ready"],
        "transcription_reason": transcription["reason"],
        "transcription_model": transcription["model"],
        "local_llm_ready": local_llm_ready,
        "local_llm_model": local_voice_llm.LOCAL_VOICE_LLM_MODEL,
        "tts": "browser-local-voices-only",
    }


@router.post("/transcribe")
async def transcribe_local_audio(request: Request) -> dict[str, str]:
    """Transcribe a short recording with a local whisper.cpp process only."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    suffix = _AUDIO_SUFFIXES.get(content_type)
    if suffix is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported recording format. Use WebM, Ogg, MP4, MP3, or WAV audio.",
        )

    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > local_speech_to_text.MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Recording exceeds the 20 MB limit.",
        )

    audio_chunks: list[bytes] = []
    received_bytes = 0
    async for chunk in request.stream():
        received_bytes += len(chunk)
        if received_bytes > local_speech_to_text.MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Recording exceeds the 20 MB limit.",
            )
        audio_chunks.append(chunk)
    if not received_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The recording is empty. Record a short message and try again.",
        )
    audio = b"".join(audio_chunks)

    language_hint = request.query_params.get("language", "auto")
    if language_hint not in {"auto", "en", "hi"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Speech language must be auto, en, or hi.",
        )

    transcript = await run_in_threadpool(
        local_speech_to_text.transcribe_with_whisper_cpp, audio, suffix, language_hint
    )
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No speech was recognized. Try again in a quieter place or edit the recording.",
        )
    return {"text": transcript, "language_hint": language_hint}


@router.post("/chat")
async def generate_local_voice_reply(request: LocalVoiceChatRequest) -> dict[str, str]:
    """Use only the configured Ollama model on loopback for voice-originated messages."""
    if not local_voice_llm.VOICE_LOCAL_ONLY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Local-only voice replies are disabled by server configuration.",
        )
    return await local_voice_llm.generate_local_reply(
        [message.model_dump() for message in request.messages],
        request.language_preference,
    )
