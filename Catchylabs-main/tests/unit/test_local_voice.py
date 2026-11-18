import subprocess
import tempfile
import wave
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import local_voice
import local_speech_to_text
import local_voice_llm


class LocalVoiceTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(local_voice.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()

    def test_status_reports_missing_local_models_without_download_or_path_disclosure(self):
        with (
            patch.object(local_speech_to_text, "_configured_binary", return_value=None),
            patch.object(local_speech_to_text, "_configured_stt_model", return_value=None),
            patch.object(local_voice_llm, "local_llm_available", new=AsyncMock(return_value=False)),
            patch.object(local_voice_llm, "VOICE_LOCAL_ONLY", True),
        ):
            response = self.client.get("/voice/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["local_only"])
        self.assertFalse(body["transcription_ready"])
        self.assertFalse(body["local_llm_ready"])
        self.assertNotIn("LOCAL_STT_MODEL_PATH", str(body))
        self.assertNotIn(tempfile.gettempdir(), str(body))

    def test_rejects_unsupported_audio_format_before_processing(self):
        with patch.object(local_speech_to_text, "transcribe_with_whisper_cpp") as transcribe:
            response = self.client.post(
                "/voice/transcribe",
                content=b"audio",
                headers={"Content-Type": "application/octet-stream"},
            )

        self.assertEqual(response.status_code, 415)
        transcribe.assert_not_called()

    def test_rejects_oversized_content_length_before_reading_audio(self):
        response = self.client.post(
            "/voice/transcribe",
            content=b"",
            headers={
                "Content-Type": "audio/webm",
                "Content-Length": str(local_speech_to_text.MAX_AUDIO_BYTES + 1),
            },
        )
        self.assertEqual(response.status_code, 413)

    def test_valid_audio_is_transcribed_with_local_processes_and_temp_files_are_removed(self):
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir) / "ggml-base.bin"
            model_path.write_bytes(b"test model fixture")

            def fake_process(arguments, **_kwargs):
                if arguments[0] == "/local/ffmpeg":
                    with wave.open(arguments[-1], "wb") as audio_file:
                        audio_file.setnchannels(1)
                        audio_file.setsampwidth(2)
                        audio_file.setframerate(16_000)
                        audio_file.writeframes(b"\0\0" * 16_000)
                    return subprocess.CompletedProcess(args=arguments, returncode=0, stdout="", stderr="")
                return subprocess.CompletedProcess(
                    args=arguments, returncode=0, stdout="नमस्ते CatchyLab", stderr=""
                )

            with (
                patch.object(local_speech_to_text, "_configured_binary", side_effect=["/local/whisper-cli", "/local/ffmpeg"]),
                patch.object(local_speech_to_text, "_configured_stt_model", return_value=model_path),
                patch.object(local_speech_to_text.subprocess, "run", side_effect=fake_process) as run,
            ):
                transcript = local_speech_to_text.transcribe_with_whisper_cpp(b"webm bytes", ".webm", "hi")

        self.assertEqual(transcript, "नमस्ते CatchyLab")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][0], "/local/ffmpeg")
        self.assertEqual(run.call_args_list[1].args[0][0], "/local/whisper-cli")
        self.assertIn("hi", run.call_args_list[1].args[0])
        audio_path = Path(run.call_args_list[0].args[0][-1]).parent / "recording.webm"
        self.assertFalse(audio_path.exists())

    def test_auto_english_and_hindi_language_hints_are_sent_to_local_stt(self):
        for language in ("auto", "en", "hi"):
            with self.subTest(language=language):
                with patch.object(
                    local_speech_to_text,
                    "transcribe_with_whisper_cpp",
                    return_value="recognized locally",
                ) as transcribe:
                    response = self.client.post(
                        f"/voice/transcribe?language={language}",
                        content=b"audio fixture",
                        headers={"Content-Type": "audio/webm"},
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(transcribe.call_args.args[2], language)

    def test_voice_chat_posts_only_to_fixed_loopback_ollama_and_returns_actual_text(self):
        response = httpx.Response(
            200,
            json={"message": {"content": "स्थानीय जवाब"}},
            request=httpx.Request("POST", local_voice_llm._OLLAMA_CHAT_URL),
        )
        with (
            patch.object(local_voice_llm, "VOICE_LOCAL_ONLY", True),
            patch.object(local_voice_llm.httpx, "AsyncClient") as client_class,
        ):
            client = client_class.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.post = AsyncMock(return_value=response)
            body = self.client.post(
                "/voice/chat",
                json={
                    "messages": [{"role": "user", "content": "नमस्ते"}],
                    "language_preference": "hindi",
                },
            )
            self.assertEqual(body.status_code, 200)
            self.assertEqual(body.json()["text"], "स्थानीय जवाब")
            self.assertEqual(client.post.await_args.args[0], "http://127.0.0.1:11434/api/chat")
            self.assertFalse(client.post.await_args.kwargs["json"]["stream"])

    def test_local_voice_chat_is_disabled_when_server_config_opts_out(self):
        with patch.object(local_voice_llm, "VOICE_LOCAL_ONLY", False):
            response = self.client.post(
                "/voice/chat",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        self.assertEqual(response.status_code, 409)
        self.assertIn("disabled", response.json()["detail"])

    def test_context_limit_and_last_user_message_are_validated(self):
        too_many_chars = [{"role": "user", "content": "a" * 6000} for _ in range(6)]
        invalid_contexts = [
            {"messages": [{"role": "assistant", "content": "Not a user turn"}]},
            {"messages": too_many_chars},
        ]
        for payload in invalid_contexts:
            with self.subTest(payload=payload):
                response = self.client.post("/voice/chat", json=payload)
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
