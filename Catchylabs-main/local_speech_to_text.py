"""Local whisper.cpp speech-to-text adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status


MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 60


def _configured_binary(variable: str, default: str) -> str | None:
    configured = os.environ.get(variable, default).strip()
    if not configured:
        return None
    candidate = Path(configured)
    if candidate.is_absolute():
        return str(candidate) if candidate.is_file() else None
    return shutil.which(configured)


def _configured_stt_model() -> Path | None:
    configured = os.environ.get("LOCAL_STT_MODEL_PATH", "").strip()
    if not configured:
        return None
    model_path = Path(configured).expanduser()
    return model_path.resolve() if model_path.is_file() else None


def local_transcription_status() -> dict[str, Any]:
    whisper_binary = _configured_binary("LOCAL_STT_BINARY", "whisper-cli")
    ffmpeg_binary = _configured_binary("LOCAL_STT_FFMPEG", "ffmpeg")
    model_path = _configured_stt_model()
    ready = bool(whisper_binary and ffmpeg_binary and model_path)
    return {
        "ready": ready,
        "reason": None if ready else "Local whisper.cpp, ffmpeg, and a downloaded multilingual model are required.",
        "model": model_path.name if model_path else None,
    }


def transcribe_with_whisper_cpp(audio: bytes, suffix: str, language: str) -> str:
    whisper_binary = _configured_binary("LOCAL_STT_BINARY", "whisper-cli")
    ffmpeg_binary = _configured_binary("LOCAL_STT_FFMPEG", "ffmpeg")
    model_path = _configured_stt_model()
    if not whisper_binary or not ffmpeg_binary or not model_path:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local transcription is not configured. Install whisper.cpp, ffmpeg, and a multilingual model on this CatchyLab backend.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="catchylab-local-voice-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / f"recording{suffix}"
            wav_path = temp_path / "recording.wav"
            input_path.write_bytes(audio)
            try:
                conversion = subprocess.run(
                    [
                        ffmpeg_binary, "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-i", str(input_path), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", "-t", str(MAX_AUDIO_SECONDS + 1), str(wav_path),
                    ],
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The recording could not be decoded locally. Try a shorter recording.",
                ) from exc
            if conversion.returncode != 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The recording format could not be decoded locally. Try another browser or recording.",
                )
            try:
                with wave.open(str(wav_path), "rb") as converted_audio:
                    duration_seconds = converted_audio.getnframes() / converted_audio.getframerate()
            except (wave.Error, OSError, ZeroDivisionError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The recording could not be decoded as audio. Try another browser or recording.",
                ) from exc
            if duration_seconds > MAX_AUDIO_SECONDS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Recordings are limited to 60 seconds.",
                )

            try:
                transcription = subprocess.run(
                    [
                        whisper_binary, "-m", str(model_path), "-f", str(wav_path),
                        "-l", language, "-nt", "-np",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=120,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Local transcription timed out. Try a shorter recording or a smaller local model.",
                ) from exc
            if transcription.returncode != 0:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The configured local speech model could not transcribe this recording.",
                )
            return transcription.stdout.strip()
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local speech runtime could not access its temporary workspace or executable.",
        ) from exc
