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

async def extract_attachment_text(attachment, executor, max_image_bytes: int = 8 * 1024 * 1024) -> str:
    """Use Groq vision to describe/read images."""
    from core.ai import groq_call
    
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        return ""

    # Reject images over size limit
    if attachment.size and attachment.size > max_image_bytes:
        log.warning(f"[vision] image too large ({attachment.size} bytes), skipping")
        return ""

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
                                "text": "Describe this image concisely. If it contains text, include it. If it's a meme, explain it.",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
        )
        
        log.debug(f"[vision] result: {result[0][:80] if result else 'empty'}")
        return f"[Image: {result[0]}]" if result and result[0] else ""
    except Exception as e:
        log.error(f"[vision] failed: {e}")
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
