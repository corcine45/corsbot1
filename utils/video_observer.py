'''utils/video_observer.py
Utility for a Discord bot to observe/watch a video.

Features:
- Download a video (or audio‑only) from a public URL using ``yt-dlp``.
- Extract frames at a configurable interval with ``opencv``.
- Run a vision model (e.g. ``transformers`` BLIP image captioning) on each frame.
- Optionally run an ASR model (e.g. ``whisper``) on the audio track.
- Return a concise summary that the bot can send as a message.

Installation requirements (once you add the file, run these commands in the project root):

```bash
! pip install yt-dlp opencv-python pillow torch torchvision transformers
! pip install -U openai-whisper  # optional for speech‑to‑text
```

The class below is deliberately side‑effect‑free – it only writes to the ``tmp`` folder inside the project and cleans up after itself.
'''\n\nimport os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple, Optional\n\nimport cv2
from PIL import Image\n\n# Vision model – BLIP image captioning (you can swap for any model that accepts a PIL.Image)
from transformers import BlipProcessor, BlipForConditionalGeneration\n\n# Optional speech‑to‑text model (requires ``whisper`` package). Import lazily so the bot works without it if not needed.
try:\n    import whisper\nexcept ImportError:\n    whisper = None\n\n\nclass VideoObserver:\n    """Utility class for a bot to *watch* a video and produce a summary.
\n    Typical usage::\n\n        observer = VideoObserver(url="https://youtu.be/abc123")\n        summary = observer.summarise()\n        await channel.send(summary)\n\n    The implementation follows three steps:\n\n    1️⃣ **Download** – uses ``yt-dlp`` to fetch the best‑quality video (or audio‑only if ``audio_only=True``).\n    2️⃣ **Frame extraction** – opens the video with OpenCV and extracts ``frame_interval`` seconds of visual content.\n    3️⃣ **Captioning** – runs a pretrained BLIP model on each extracted frame and aggregates the captions.\n\n    If ``audio_only`` is ``True`` AND the optional ``whisper`` library is installed, the audio track is transcribed instead of (or in addition to) visual captioning.
\n    All temporary files are stored under ``self.work_dir`` and removed when the instance is deleted.
    """\n\n    def __init__(\n        self,
        url: str,
        work_dir: Optional[Path] = None,
        frame_interval: int = 5,
        max_frames: int = 10,
        audio_only: bool = False,
        language: str = "en",
    ) -> None:\n        self.url = url\n        self.frame_interval = max(1, frame_interval)  # seconds between extracted frames\n        self.max_frames = max_frames\n        self.audio_only = audio_only\n        self.language = language\n\n        # Create an isolated temporary directory for this run\n        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="video_observer_"))\n        self.video_path = self.work_dir / f"{uuid.uuid4()}.mp4"\n        self.audio_path = self.work_dir / f"{uuid.uuid4()}.wav"\n\n        # Initialise the vision model once (takes a few seconds on first load)\n        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")\n        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")\n\n    def __del__(self) -> None:\n        # Clean up the temp folder when the object is garbage‑collected\n        try:\n            if self.work_dir.exists():\n                for child in self.work_dir.iterdir():\n                    child.unlink(missing_ok=True)\n                self.work_dir.rmdir()\n        except Exception:\n            pass  # best‑effort cleanup – failures are harmless in a short‑lived bot process\n\n    # ---------------------------------------------------------------------\n    # Step 1 – download\n    # ---------------------------------------------------------------------\n    def _download(self) -> None:\n        """Download the video (or audio) using ``yt-dlp``.
\n        The command is executed in a subprocess with ``-f best`` for full video or ``-x`` for audio‑only.
        Errors raise ``RuntimeError``.
        """\n        if self.audio_only:\n            cmd = [\n                "yt-dlp",
                "-f", "bestaudio",
                "-x", "--audio-format", "wav",
                "-o", str(self.audio_path),
                self.url,
            ]\n        else:\n            cmd = [\n                "yt-dlp",
                "-f", "best",
                "-o", str(self.video_path),
                self.url,
            ]\n\n        result = subprocess.run(cmd, capture_output=True, text=True)\n        if result.returncode != 0:\n            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")\n\n    # ---------------------------------------------------------------------\n    # Step 2 – frame extraction (video path only)\n    # ---------------------------------------------------------------------\n    def _extract_frames(self) -> List[Path]:\n        """Return a list of file paths for the extracted frames.
\n        Frames are spaced by ``self.frame_interval`` seconds, up to ``self.max_frames``.
        """\n        cap = cv2.VideoCapture(str(self.video_path))\n        if not cap.isOpened():\n            raise RuntimeError("Unable to open downloaded video file")\n\n        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 if unknown\n        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n        duration = frame_count / fps\n\n        frames: List[Path] = []\n        timestamps = [i for i in range(0, int(duration), self.frame_interval)][: self.max_frames]\n\n        for ts in timestamps:\n            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)\n            ret, frame = cap.read()\n            if not ret:\n                continue\n            img_path = self.work_dir / f"frame_{ts}.jpg"\n            cv2.imwrite(str(img_path), frame)\n            frames.append(img_path)\n\n        cap.release()\n        return frames\n\n    # ---------------------------------------------------------------------\n    # Step 3 – captioning (visual)\n    # ---------------------------------------------------------------------\n    def _caption_frames(self, frames: List[Path]) -> List[str]:\n        """Run BLIP captioning on each frame and return the raw captions."""\n        captions: List[str] = []\n        for fp in frames:\n            image = Image.open(fp).convert("RGB")\n            inputs = self.processor(images=image, return_tensors="pt")\n            out = self.caption_model.generate(**inputs)\n            caption = self.processor.decode(out[0], skip_special_tokens=True)\n            captions.append(caption)\n        return captions\n\n    # ---------------------------------------------------------------------\n    # Optional audio transcription (requires ``whisper``)\n    # ---------------------------------------------------------------------\n    def _transcribe_audio(self) -> Optional[str]:\n        if not whisper:\n            return None  # whisper not installed – silently skip\n        if not self.audio_path.exists():\n            raise RuntimeError("Audio file not found for transcription")\n        model = whisper.load_model("base")  # fast, reasonable quality\n        result = model.transcribe(str(self.audio_path), language=self.language)\n        return result.get("text")\n\n    # ---------------------------------------------------------------------\n    # Public API – high level summary\n    # ---------------------------------------------------------------------\n    def summarise(self) -> str:\n        """Download the media, process it, and return a concise summary.
\n        The summary combines visual captions (if a video) and an optional audio transcript.
        """\n        # 1️⃣ Download\n        self._download()\n\n        captions: List[str] = []\n        transcript: Optional[str] = None\n\n        if self.audio_only:\n            # Only audio – transcribe (if whisper is present)\n            transcript = self._transcribe_audio()\n        else:\n            # Full video – extract frames and generate captions\n            frames = self._extract_frames()\n            if frames:\n                captions = self._caption_frames(frames)\n\n        # Build a short human‑readable summary\n        parts: List[str] = []\n        if captions:\n            # Collapse similar captions – naive deduplication\n            unique = []\n            for c in captions:\n                if c not in unique:\n                    unique.append(c)\n            parts.append("Visual highlights: " + "; ".join(unique))\n        if transcript:\n            # Only keep the first couple of sentences to avoid flooding chat\n            excerpt = transcript.strip().split(".")[:2]\n            parts.append("Audio transcript (first 2 sentences): " + ". ".join(excerpt).strip() + ".")\n\n        if not parts:\n            return "I couldn't extract any visual or audio information from the provided link."
        return " | ".join(parts)\n\n    # ---------------------------------------------------------------------\n    # Convenience helper – call from an async command handler\n    # ---------------------------------------------------------------------\n    async def async_summarise(self) -> str:\n        """Wrap ``summarise`` so it can be awaited in an ``async`` Discord command.
\n        The heavy work runs in a thread pool to avoid blocking the event loop.
        """\n        import asyncio\n        loop = asyncio.get_running_loop()\n        return await loop.run_in_executor(None, self.summarise)\n\n# Example usage (for local testing)\nif __name__ == "__main__":\n    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # replace with any public video\n    observer = VideoObserver(url=test_url, frame_interval=10, max_frames=5)\n    print(observer.summarise())\n"