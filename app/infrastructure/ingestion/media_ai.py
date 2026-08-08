"""OpenRouter / Groq audio STT and OpenRouter image vision via httpx.

OpenRouter STT: POST https://openrouter.ai/api/v1/audio/transcriptions with a
JSON body `{"model": ..., "input_audio": {"data": <raw base64>, "format": ...}}`.
Note: OpenRouter requires a >=$0.50 account balance for any audio request (402
otherwise). GroqSTT uses Groq's OpenAI-compatible (multipart) transcriptions
API, which has a free tier and no such balance requirement.

Vision: standard OpenAI-style chat completion with an `image_url` data URI,
asking the model to both describe the image and transcribe any readable text
(basic OCR for charts and screenshots).
"""

from __future__ import annotations

import base64

import httpx

from app.application.ingestion.pipeline import VisionAnalyzer
from app.application.ingestion.types import FileData
from app.core.logging import get_logger

logger = get_logger(__name__)

_AUDIO_TRANSCRIPTIONS_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

_AUDIO_FORMATS = ("ogg", "mp3", "wav", "m4a", "webm", "opus", "flac", "aac", "mp4")
_IMAGE_MIME = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")

_VISION_PROMPT = (
    "This is an image a user sent to their financial assistant. "
    "Describe what the image shows in plain text. If it contains a chart, "
    "table, report page, or screenshot with readable text, transcribe the "
    "relevant numbers and labels exactly (OCR). Be factual and complete; "
    "do not interpret the meaning."
)


_GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqSTT:
    """Free speech-to-text via Groq's OpenAI-compatible transcriptions API.

    Groq's Whisper endpoints have a free tier (rate-limited); no OpenRouter
    balance requirement. The API is OpenAI-style multipart (`file` + `model`),
    so this is a plain httpx call with no extra SDK dependency.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-large-v3-turbo",
        timeout_seconds: float = 90.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def transcribe(self, source: FileData) -> str:
        fmt = OpenRouterMediaAI._audio_format(source)
        filename = f"voice.{fmt}"
        mime = source.mime_type or "application/octet-stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await self._http.post(
                _GROQ_TRANSCRIPTIONS_URL,
                data={"model": self._model},
                headers=headers,
                files={"file": (filename, source.raw, mime)},
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Groq STT request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"Groq STT error {response.status_code}: {response.text[:200]}")
        payload = response.json()
        text = (payload.get("text") or "").strip()
        if not text:
            raise RuntimeError("Groq STT returned empty transcription")
        return text


class OpenRouterMediaAI:
    def __init__(
        self,
        api_key: str,
        *,
        stt_model: str = "openai/whisper-1",
        vision_model: str = "openai/gpt-4o-mini",
        timeout_seconds: float = 90.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._stt_model = stt_model
        self._vision_model = vision_model
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def transcribe(self, data: FileData) -> str:
        fmt = self._audio_format(data)
        body = {
            "model": self._stt_model,
            "input_audio": {
                "data": base64.b64encode(data.raw).decode("ascii"),
                "format": fmt,
            },
        }
        payload = await self._post(_AUDIO_TRANSCRIPTIONS_URL, body)
        text = (payload.get("text") or "").strip()
        if not text:
            raise RuntimeError("STT returned empty transcription")
        return text

    async def describe(self, data: FileData) -> str:
        mime = self._image_mime(data)
        data_uri = f"data:{mime};base64," + base64.b64encode(data.raw).decode("ascii")
        body = {
            "model": self._vision_model,
            "max_tokens": 1500,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        payload = await self._post(_CHAT_URL, body)
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("vision returned empty description")
        return text

    async def _post(self, url: str, body: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._http.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"media AI request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"media AI error {response.status_code}: {response.text[:200]}")
        return response.json()

    @staticmethod
    def _audio_format(data: FileData) -> str:
        mime = (data.mime_type or "").lower()
        if "ogg" in mime or mime == "audio/opus":
            return "ogg"
        if "mp3" in mime or mime == "audio/mpeg":
            return "mp3"
        if "wav" in mime:
            return "wav"
        if "m4a" in mime or "mp4" in mime:
            return "m4a"
        if "webm" in mime:
            return "webm"
        if "flac" in mime:
            return "flac"
        name = (data.filename or "").lower()
        for fmt in _AUDIO_FORMATS:
            if name.endswith(f".{fmt}"):
                return fmt
        return "ogg"

    @staticmethod
    def _image_mime(data: FileData) -> str:
        mime = (data.mime_type or "").lower()
        if mime in _IMAGE_MIME:
            return mime
        name = (data.filename or "").lower()
        for ext, m in (
            (".png", "image/png"),
            (".webp", "image/webp"),
            (".gif", "image/gif"),
            (".bmp", "image/bmp"),
        ):
            if name.endswith(ext):
                return m
        return "image/jpeg"


class GroqVision:
    """Free image describe/OCR via Groq's OpenAI-compatible chat completions.

    Same wire shape as OpenRouterMediaAI.describe (model, messages with an
    `image_url` data URI), so it is a drop-in VisionAnalyzer. Free tier, no
    OpenRouter balance requirement.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "meta-llama/llama-3.2-11b-vision-preview",
        timeout_seconds: float = 60.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def describe(self, data: FileData) -> str:
        mime = OpenRouterMediaAI._image_mime(data)
        data_uri = f"data:{mime};base64," + base64.b64encode(data.raw).decode("ascii")
        body = {
            "model": self._model,
            "max_tokens": 1500,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._http.post(_GROQ_CHAT_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Groq vision request failed: {exc}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"Groq vision error {response.status_code}: {response.text[:200]}")
        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("Groq vision returned empty description")
        return text


class VisionFallback:
    """Tries the primary vision provider and falls back to a secondary one.

    OpenRouter routes 402 when the account balance is exhausted; the fallback
    keeps image describe/OCR working on the free Groq tier. When both fail the
    secondary's error is re-raised so the ingestion pipeline records it
    gracefully.
    """

    def __init__(self, primary: VisionAnalyzer, fallback: VisionAnalyzer) -> None:
        self._primary = primary
        self._fallback = fallback

    async def describe(self, data: FileData) -> str:
        try:
            return await self._primary.describe(data)
        except Exception as primary_exc:  # noqa: BLE001 - provider failures switch providers
            logger.warning("vision_primary_failed_using_fallback", error=str(primary_exc))
        return await self._fallback.describe(data)


__all__ = ["GroqSTT", "GroqVision", "OpenRouterMediaAI", "VisionFallback"]
