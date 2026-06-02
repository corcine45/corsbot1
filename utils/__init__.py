"""
Utility functions for Discord interactions and string processing.
"""

import hashlib
import re
import time
import random
import asyncio
import aiohttp
import base64
import logging
from typing import Optional

from config import settings

log = logging.getLogger("corsbot.utils")

# ────────────────────────────────────────────────────────────────────────────────
# IMAGE ANALYSIS PROMPTS
# ────────────────────────────────────────────────────────────────────────────────

# Master prompt for general image analysis — used as the base for all image types
_IMAGE_ANALYSIS_PROMPT = (
    "You are analyzing an image shared in a Discord chat. Provide a concise, natural description "
    "that captures what's happening and why someone might share it. "
    "Focus on: (1) the main subject/action, (2) any readable text (quote exactly), "
    "(3) the vibe/context (meme, screenshot, photo, art, etc.), "
    "(4) any notable details that explain why it's funny, interesting, or worth sharing. "
    "Be specific — avoid vague descriptions like 'a picture of a person.' "
    "If something is unclear or ambiguous, say so. "
    "Do NOT invent names, franchises, celebrities, or identities unless there's clear visual evidence "
    "(like text, logos, or unmistakable features)."
)

# Specialized prompts for different image types
_MEME_PROMPT = (
    "Analyze this meme image. Explain: "
    "(1) What text is on the image (quote it exactly), "
    "(2) The format/template if recognizable (e.g., 'Drake meme', 'Distracted boyfriend', 'woman yelling at cat'), "
    "(3) What the joke means — what's being compared, mocked, or referenced, "
    "(4) The tone (self-deprecating, absurdist, relatable, etc.). "
    "If it's a niche meme format you don't recognize, describe the visual layout and text instead of guessing."
)

_SCREENSHOT_PROMPT = (
    "Analyze this screenshot. Identify: "
    "(1) What app/website/game is shown (look at UI elements, logos, layout), "
    "(2) What's happening in the screenshot (a message, error, score, post, etc.), "
    "(3) Any readable text that's important to understanding the context, "
    "(4) Why someone might share this (flexing, complaining, asking for help, humor, etc.). "
    "Read all visible text carefully and include the key parts."
)

_PHOTO_PROMPT = (
    "Analyze this photograph. Describe: "
    "(1) The main subject(s) and what they're doing, "
    "(2) The setting/location and lighting/mood, "
    "(3) Any notable details (expressions, objects, text in the background), "
    "(4) The overall vibe (casual selfie, professional shot, candid, aesthetic, etc.). "
    "If it appears to be a pet photo, focus on the animal's appearance and expression."
)

_ART_PROMPT = (
    "Analyze this artwork/illustration. Describe: "
    "(1) The medium/style (digital art, anime, pixel art, traditional, etc.), "
    "(2) The subject and composition, "
    "(3) The color palette and mood, "
    "(4) Any signature, watermark, or artist credit visible, "
    "(5) Any text within the image. "
    "Note if it appears to be fan art of a known style, but don't guess specific characters unless certain."
)

# Keywords to detect image type from conversation context
_MEME_CONTEXT_KEYWORDS = {
    "meme", "funny", "lol", "lmao", "joke", "humor", "comedy", "dank", "shitpost",
    "relatable", "mood", "template", "format", "caption",
}

_SCREENSHOT_CONTEXT_KEYWORDS = {
    "screenshot", "ss", "look at this", "check this", "error", "bug", "glitch",
    "text", "message", "chat", "post", "tweet", "comment", "score", "stats",
    "look", "see this", "check out",
}

_ART_CONTEXT_KEYWORDS = {
    "art", "drawing", "artwork", "illustration", "fanart", "fan art", "digital art",
    "sketch", "painting", "anime", "manga", "oc", "original character",
    "pixel art", "icon", "banner", "wallpaper",
}

_PHOTO_CONTEXT_KEYWORDS = {
    "photo", "picture", "pic", "selfie", "fit check", "outfit", "pet", "cat", "dog",
    "food", "view", "sunset", "travel", "vacation", "my room", "setup", "battlestation",
    "look at my", "check my", "my new",
}


# ────────────────────────────────────────────────────────────────────────────────
# MESSAGE SENDING
# ────────────────────────────────────────────────────────────────────────────────

async def send_reply(channel, text: str, limit: int = 2000):
    """Split long messages and send them, respecting Discord's 2000 char limit."""
    while len(text) > limit:
        split = text.rfind("\n", 0, limit)
        if split == -1:
            split = limit
        await channel.send(text[:split])
        text = text[split:].lstrip("\n")
    if text:
        await channel.send(text)


async def send_interaction(interaction, text: str, ephemeral: bool = True):
    """Send a slash command response, splitting if needed."""
    if len(text) <= 2000:
        await interaction.response.send_message(text, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(text[:2000], ephemeral=ephemeral)
        rest = text[2000:]
        while rest:
            await interaction.followup.send(rest[:2000], ephemeral=ephemeral)
            rest = rest[2000:]


# ────────────────────────────────────────────────────────────────────────────────
# MENTION RESOLUTION
# ────────────────────────────────────────────────────────────────────────────────

def resolve_mentions_in_reply(reply: str, guild) -> str:
    """Replace @name mentions with Discord mention format."""
    if not guild:
        return reply
    
    def replace_mention(match):
        name = match.group(1).lower()
        for member in guild.members:
            if member.display_name.lower() == name or member.name.lower() == name:
                return f"<@{member.id}>"
        return match.group(0)
    
    return re.sub(r"@([\w\s]+)", replace_mention, reply)


# ────────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ────────────────────────────────────────────────────────────────────────────────


def _detect_image_type_from_context(message_content: str = "") -> str:
    """
    Detect the likely image type based on conversation context keywords.
    Returns: 'meme', 'screenshot', 'art', 'photo', or 'general'
    """
    if not message_content:
        return "general"
    
    lower = message_content.lower()
    words = lower.split()
    
    # Score each category
    scores = {
        "meme": 0,
        "screenshot": 0,
        "art": 0,
        "photo": 0,
    }
    
    for keyword in _MEME_CONTEXT_KEYWORDS:
        if keyword in lower:
            scores["meme"] += 1
    for keyword in _SCREENSHOT_CONTEXT_KEYWORDS:
        if keyword in lower:
            scores["screenshot"] += 1
    for keyword in _ART_CONTEXT_KEYWORDS:
        if keyword in lower:
            scores["art"] += 1
    for keyword in _PHOTO_CONTEXT_KEYWORDS:
        if keyword in lower:
            scores["photo"] += 1
    
    # Check for multi-word phrases (higher confidence)
    for phrase in ["fit check", "look at this", "check this", "check out", 
                    "look at my", "check my", "my new", "original character"]:
        if phrase in lower:
            if phrase in ("fit check",):
                scores["photo"] += 2
            elif phrase in ("look at this", "check this", "check out"):
                scores["screenshot"] += 2
            elif phrase in ("look at my", "check my", "my new"):
                scores["photo"] += 2
    
    max_score = max(scores.values())
    if max_score == 0:
        return "general"
    
    # Return the highest scoring category
    for category, score in scores.items():
        if score == max_score:
            return category
    
    return "general"


def _get_prompt_for_image_type(image_type: str) -> str:
    """Get the appropriate analysis prompt for the detected image type."""
    prompts = {
        "meme": _MEME_PROMPT,
        "screenshot": _SCREENSHOT_PROMPT,
        "art": _ART_PROMPT,
        "photo": _PHOTO_PROMPT,
        "general": _IMAGE_ANALYSIS_PROMPT,
    }
    return prompts.get(image_type, _IMAGE_ANALYSIS_PROMPT)


async def extract_attachment_text(
    attachment, 
    executor, 
    max_image_bytes: int = 8 * 1024 * 1024,
    message_content: str = "",
    conversation_context: str = "",
) -> str:
    """
    Use vision AI to describe/read images with context-aware analysis.
    
    Args:
        attachment: Discord attachment object
        executor: Thread pool executor for async operations
        max_image_bytes: Maximum allowed image size
        message_content: The text message that accompanied the image (for context)
        conversation_context: Recent conversation history (for better type detection)
    
    Returns:
        Formatted string like "[Image: description]" or empty string on failure
    """
    from core.ai import groq_call
    
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        return ""

    # Reject images over size limit
    if attachment.size and attachment.size > max_image_bytes:
        log.warning(f"[vision] image too large ({attachment.size} bytes), skipping")
        return ""

    # Detect image type from context for better analysis
    full_context = f"{message_content}\n{conversation_context}"
    image_type = _detect_image_type_from_context(full_context)
    analysis_prompt = _get_prompt_for_image_type(image_type)
    
    log.debug(f"[vision] detected image type: {image_type}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                if response.status != 200:
                    log.warning(f"[vision] failed to download image: {response.status}")
                    return ""
                
                # Double-check Content-Length if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_image_bytes:
                    log.warning(f"[vision] Content-Length too large, skipping")
                    return ""
                
                data = await response.read()

        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{attachment.content_type};base64,{b64}"

        loop = asyncio.get_running_loop()
        
        # Try primary vision model (Groq)
        try:
            result = await loop.run_in_executor(
                executor,
                lambda: groq_call(
                    settings.vision_model,
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                                {
                                    "type": "text",
                                    "text": analysis_prompt,
                                },
                            ],
                        }
                    ],
                    max_tokens=350,
                    timeout=15,
                )
            )
        except Exception as groq_error:
            # Fallback to Gemini vision if available
            log.warning(f"[vision] Groq vision failed, trying Gemini fallback: {groq_error}")
            if settings.gemini_api_key:
                result = await _gemini_vision_call(data_url, attachment.content_type, analysis_prompt)
            else:
                raise groq_error
        
        description = result[0] if result and result[0] else ""
        
        if description:
            # Add image type tag for the AI to understand context better
            type_tag = f"[Image: {image_type}] " if image_type != "general" else "[Image: "
            return f"{type_tag}{description}]"
        
        return ""
        
    except Exception as e:
        log.error(f"[vision] failed: {e}")
        return ""


async def _gemini_vision_call(data_url: str, content_type: str, prompt: str) -> tuple:
    """
    Fallback vision analysis using Gemini API.
    Returns (description, tokens_used) tuple.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": content_type,
                    "data": data_url.split(",", 1)[1] if "," in data_url else data_url
                }}
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 350,
            "temperature": 0.4,
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            params={"key": settings.gemini_api_key},
            json=payload,
            timeout=15
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"Gemini vision {response.status}: {await response.text()}")
            
            data = await response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini vision returned no candidates: {data}")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()
            
            if not text:
                raise RuntimeError("Gemini vision returned empty response")
            
            usage = data.get("usageMetadata", {})
            tokens = usage.get("totalTokenCount", 0)
            
            return text, tokens


# ────────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ────────────────────────────────────────────────────────────────────────────────

_VIDEO_ANALYSIS_PROMPT = (
    "Analyze these frames from a video shared in Discord. Provide a concise description covering:\n"
    "(1) What's happening in the video — the main action, subject, or content,\n"
    "(2) The type of video (meme, clip, gameplay, IRL footage, animation, etc.),\n"
    "(3) Any readable text, subtitles, or on-screen captions,\n"
    "(4) The overall vibe or why someone might share this.\n"
    "If the frames show different moments, describe the progression. "
    "If it's a static or near-static video, note that. "
    "Do NOT invent names, franchises, or identities unless there's clear visual evidence."
)

# Maximum video size for analysis (default 25MB)
MAX_VIDEO_BYTES = 25 * 1024 * 1024

# Number of frames to extract from video
VIDEO_FRAME_COUNT = 4

# Video content type prefixes
_VIDEO_CONTENT_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", 
    "video/avi", "video/mpeg", "video/x-matroska",
}


def _detect_video_type_from_context(message_content: str = "") -> str:
    """Detect likely video type from conversation context."""
    if not message_content:
        return "general"
    
    lower = message_content.lower()
    
    if any(kw in lower for kw in ("meme", "funny", "lol", "lmao", "dank", "shitpost")):
        return "meme"
    if any(kw in lower for kw in ("gameplay", "gaming", "game", "playthrough", "walkthrough")):
        return "gameplay"
    if any(kw in lower for kw in ("tiktok", "reel", "shorts", "short video")):
        return "social_media"
    if any(kw in lower for kw in ("music", "mv", "music video", "official video", "lyric video")):
        return "music"
    if any(kw in lower for kw in ("movie", "film", "scene", "episode", "show", "trailer")):
        return "film_tv"
    
    return "general"


def _extract_frames_from_video(data: bytes, content_type: str) -> list[bytes]:
    """
    Extract key frames from video data.
    
    This is a lightweight implementation that extracts frames from specific
    time positions. For production use with heavy video processing, consider
    using ffmpeg or a dedicated video processing service.
    
    Returns a list of frame data (as JPEG bytes).
    """
    try:
        # Try to use PIL for frame extraction
        from PIL import Image
        import io
        
        # Load video into a BytesIO buffer
        video_buffer = io.BytesIO(data)
        
        # For a simple implementation without ffmpeg, we'll extract frames
        # by trying to find keyframes in the video data. However, proper
        # video frame extraction requires ffmpeg or similar.
        
        # Since we can't reliably extract frames without ffmpeg, we'll use
        # a fallback approach: send the video file info and let the vision
        # model know it's a video, or skip video analysis if no ffmpeg.
        
        log.debug("[video] PIL available but ffmpeg-python not installed, using metadata-only approach")
        return []
        
    except ImportError:
        log.debug("[video] PIL not available for frame extraction")
        return []
    except Exception as e:
        log.debug(f"[video] frame extraction failed: {e}")
        return []


async def extract_video_description(
    attachment,
    executor,
    message_content: str = "",
    conversation_context: str = "",
    max_video_bytes: int = MAX_VIDEO_BYTES,
) -> str:
    """
    Analyze a video attachment and return a description.
    
    This function attempts to extract key frames from the video and analyze them.
    If frame extraction fails, it falls back to analyzing based on file metadata
    and context.
    
    Args:
        attachment: Discord attachment object
        executor: Thread pool executor for async operations
        message_content: The text message that accompanied the video
        conversation_context: Recent conversation history
        max_video_bytes: Maximum allowed video size
    
    Returns:
        Formatted string like "[Video: description]" or empty string on failure
    """
    from core.ai import groq_call
    
    if not attachment.content_type or attachment.content_type not in _VIDEO_CONTENT_TYPES:
        return ""
    
    # Check size limit
    if attachment.size and attachment.size > max_video_bytes:
        log.warning(f"[video] video too large ({attachment.size} bytes), skipping")
        return ""
    
    # Detect video type from context
    full_context = f"{message_content}\n{conversation_context}"
    video_type = _detect_video_type_from_context(full_context)
    
    log.debug(f"[video] detected type: {video_type}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                if response.status != 200:
                    log.warning(f"[video] failed to download: {response.status}")
                    return ""
                
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_video_bytes:
                    log.warning(f"[video] Content-Length too large, skipping")
                    return ""
                
                data = await response.read()
        
        # Try to extract frames
        frames = _extract_frames_from_video(data, attachment.content_type)
        
        loop = asyncio.get_running_loop()
        
        if frames:
            # Analyze extracted frames using vision
            # Encode frames as base64 and send multiple images
            frame_data_urls = []
            for frame in frames:
                b64 = base64.b64encode(frame).decode("utf-8")
                frame_data_urls.append(f"data:image/jpeg;base64,{b64}")
            
            # Build multi-image request
            content_parts = []
            for frame_url in frame_data_urls[:VIDEO_FRAME_COUNT]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": frame_url},
                })
            content_parts.append({
                "type": "text",
                "text": _VIDEO_ANALYSIS_PROMPT,
            })
            
            try:
                result = await loop.run_in_executor(
                    executor,
                    lambda: groq_call(
                        settings.vision_model,
                        [{"role": "user", "content": content_parts}],
                        max_tokens=400,
                        timeout=20,
                    )
                )
            except Exception as e:
                log.warning(f"[video] vision analysis failed: {e}")
                result = None
            
            if result and result[0]:
                description = result[0]
                type_tag = f"[Video: {video_type}] " if video_type != "general" else "[Video: "
                return f"{type_tag}{description}]"
        
        # Fallback: Use file metadata and context to provide a basic description
        file_extension = attachment.filename.split(".")[-1].lower() if attachment.filename else "video"
        size_kb = attachment.size / 1024 if attachment.size else 0
        size_mb = size_kb / 1024
        
        type_descriptions = {
            "meme": "a meme video",
            "gameplay": "gameplay footage",
            "social_media": "a social media clip",
            "music": "a music video",
            "film_tv": "a movie/TV clip",
            "general": f"a .{file_extension} video file",
        }
        
        type_desc = type_descriptions.get(video_type, type_descriptions["general"])
        size_hint = f" ({size_mb:.1f}MB)" if size_mb > 0.5 else ""
        
        # Add context-based hints
        context_hints = []
        if message_content:
            context_hints.append(f"Shared with message: '{message_content[:50]}'")
        
        hint_str = " " + " ".join(context_hints) if context_hints else ""
        
        return f"[Video: {type_desc}{size_hint}{hint_str}]"
        
    except Exception as e:
        log.error(f"[video] failed: {e}")
        return ""


# ────────────────────────────────────────────────────────────────────────────────
# MEMORY DETECTION
# ────────────────────────────────────────────────────────────────────────────────

def is_explicit_memory_request(text: str, memory_triggers: tuple) -> bool:
    """Check if the message is explicitly asking about stored memories."""
    lower = text.lower()
    return any(trigger in lower for trigger in memory_triggers)


# ────────────────────────────────────────────────────────────────────────────────
# RESPONSE CACHING
# ────────────────────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe TTL cache with max size (LRU eviction).

    Used for: responses, GIF URLs, session state, search results.
    Evicts the oldest entry when max_size is reached.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache: dict[str, tuple[object, float]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def get(self, key: str):
        entry = self._cache.get(key)
        if not entry:
            return None
        value, ts = entry
        if time.time() - ts > self.ttl:
            self._cache.pop(key, None)
            return None
        # Move to end (LRU)
        self._cache[key] = (value, ts)
        return value

    def set(self, key: str, value):
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Backwards-compatible alias
ResponseCache = TTLCache


def build_response_cache_key(
    thread_id: str,
    content: str,
    memory: str = "",
    relationships: str = "",
    web_context: str = "",
    feedback_context: str = "",
    history_context: str = "",
) -> str:
    """Build a cache key from message and context."""
    payload = "\n".join([
        thread_id,
        content.strip().lower(),
        memory or "",
        relationships or "",
        web_context or "",
        feedback_context or "",
        history_context or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ────────────────────────────────────────────────────────────────────────────────
# QUICK REPLIES
# ────────────────────────────────────────────────────────────────────────────────

def get_quick_reply(text: str, quick_replies: dict) -> Optional[str]:
    """Match text against quick reply patterns and return a random response."""
    normalized = text.lower().strip().rstrip("!?.")
    for patterns, replies in quick_replies.items():
        if normalized in patterns:
            return random.choice(replies)
    return None
