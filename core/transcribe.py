"""
Audio transcription via Groq Whisper.
Handles voice messages and audio attachments.
"""

import tempfile
from pathlib import Path

from core.ai import client_ai
from core.logger import get_logger

log = get_logger("corsbot.transcribe")

AUDIO_EXTENSIONS = {".ogg", ".oga", ".mp3", ".wav", ".m4a", ".webm", ".flac", ".opus"}
AUDIO_CONTENT_PREFIXES = ("audio/", "video/ogg")


def is_audio_attachment(filename: str, content_type: str | None) -> bool:
    if content_type and any(content_type.startswith(p) for p in AUDIO_CONTENT_PREFIXES):
        return True
    return Path(filename or "").suffix.lower() in AUDIO_EXTENSIONS


async def transcribe_audio_bytes(data: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe raw audio bytes. Returns transcript or empty string on failure."""
    if not data:
        return ""

    suffix = Path(filename).suffix or ".ogg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                result = client_ai.audio.transcriptions.create(
                    file=(filename, audio_file.read()),
                    model="whisper-large-v3",
                )
            text = (getattr(result, "text", None) or "").strip()
            if text:
                log.debug("transcribe_ok", chars=len(text))
            return text
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        log.warning("transcribe_failed", error=str(e))
        return ""


async def transcribe_discord_attachment(attachment) -> str:
    """Download and transcribe a Discord attachment."""
    if not is_audio_attachment(attachment.filename, attachment.content_type):
        return ""
    try:
        data = await attachment.read()
        return await transcribe_audio_bytes(data, attachment.filename)
    except Exception as e:
        log.warning("transcribe_attachment_failed", error=str(e))
        return ""
