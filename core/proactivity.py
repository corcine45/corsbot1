"""
Wire-up layer for contextual proactivity (presence.py triggers).
"""

import time

import discord

from core.db import get_last_message_age, get_user_last_channel, touch_user_channel
from core.logger import get_logger
from core.presence import (
    describe_activity,
    get_proactive_context,
    should_be_proactive,
)

log = get_logger("corsbot.proactivity")

_EMPATHY_EMOTIONS = {
    "depressed",
    "anxious",
    "lonely",
    "venting",
    "frustrated",
    "angry",
    "sad",
    "melancholy",
}


def build_context(
    user_id: int,
    *,
    activity: str = "",
    emotion: str = "",
    voice_status: str | None = None,
) -> dict:
    base = get_proactive_context(user_id)
    patterns = base.get("presence_patterns", {})
    return {
        "activity": activity,
        "emotion": emotion,
        "voice_status": voice_status,
        "time_hour": time.localtime().tm_hour,
        "last_message_age": get_last_message_age(user_id),
        "presence_patterns": patterns,
    }


async def deliver_proactive(client: discord.Client, user_id: int, message: str) -> bool:
    """Send a proactive ping to the user's last bot channel, or DM."""
    channel_id, guild_id = get_user_last_channel(user_id)
    channel = client.get_channel(int(channel_id)) if channel_id else None

    if channel and hasattr(channel, "send"):
        try:
            await channel.send(f"<@{user_id}> {message}")
            log.info("proactive_sent", user_id=user_id, channel_id=channel_id)
            return True
        except (discord.Forbidden, discord.HTTPException) as e:
            log.debug("proactive_channel_failed", user_id=user_id, error=str(e))

    try:
        user = client.get_user(user_id) or await client.fetch_user(user_id)
        if user:
            await user.send(message)
            log.info("proactive_dm_sent", user_id=user_id)
            return True
    except (discord.Forbidden, discord.HTTPException) as e:
        log.debug("proactive_dm_failed", user_id=user_id, error=str(e))

    return False


async def maybe_proactive(
    client: discord.Client,
    member: discord.Member,
    *,
    activity: str = "",
    emotion: str = "",
    voice_status: str | None = None,
) -> bool:
    if member.bot:
        return False

    context = build_context(
        member.id,
        activity=activity,
        emotion=emotion,
        voice_status=voice_status,
    )
    message = should_be_proactive(str(member.id), context)
    if not message:
        return False

    return await deliver_proactive(client, member.id, message)


def note_user_interaction(user_id: int, channel_id: int, guild_id: int | None):
    touch_user_channel(user_id, channel_id, guild_id)


def should_record_emotional_end(emotion_state: str | None, route_name: str | None) -> bool:
    if route_name == "empathy":
        return True
    return bool(emotion_state and emotion_state in _EMPATHY_EMOTIONS)
