"""
Per-server settings: personality, memory isolation, opt-out channels.
"""

import json
import time

from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.guild_settings")

DEFAULT_SETTINGS = {
    "personality": "",
    "memory_isolated": 0,
    "opt_out_channels": "[]",
}


def get_guild_settings(guild_id: int | None) -> dict:
    if not guild_id:
        return dict(DEFAULT_SETTINGS)
    _, cursor = get_db()
    cursor.execute(
        "SELECT personality, memory_isolated, opt_out_channels FROM guild_settings WHERE guild_id=?",
        (str(guild_id),),
    )
    row = cursor.fetchone()
    if not row:
        return dict(DEFAULT_SETTINGS)
    return {
        "personality": row[0] or "",
        "memory_isolated": bool(row[1]),
        "opt_out_channels": row[2] or "[]",
    }


def update_guild_settings(
    guild_id: int,
    *,
    personality: str | None = None,
    memory_isolated: bool | None = None,
    opt_out_channels: list[int] | None = None,
):
    current = get_guild_settings(guild_id)
    if personality is not None:
        current["personality"] = personality
    if memory_isolated is not None:
        current["memory_isolated"] = 1 if memory_isolated else 0
    if opt_out_channels is not None:
        current["opt_out_channels"] = json.dumps(
            [str(c) for c in opt_out_channels]
        )

    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO guild_settings (guild_id, personality, memory_isolated, opt_out_channels, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(guild_id) DO UPDATE SET
               personality=excluded.personality,
               memory_isolated=excluded.memory_isolated,
               opt_out_channels=excluded.opt_out_channels,
               updated_at=excluded.updated_at""",
        (
            str(guild_id),
            current["personality"],
            current["memory_isolated"],
            current["opt_out_channels"],
            time.time(),
        ),
    )
    conn.commit()
    log.info("guild_settings_updated", guild_id=guild_id)


def is_channel_opted_out(guild_id: int | None, channel_id: int | None) -> bool:
    if not guild_id or not channel_id:
        return False
    settings = get_guild_settings(guild_id)
    try:
        channels = json.loads(settings.get("opt_out_channels") or "[]")
    except json.JSONDecodeError:
        return False
    return str(channel_id) in {str(c) for c in channels}


def toggle_channel_opt_out(guild_id: int, channel_id: int, opt_out: bool) -> list[str]:
    settings = get_guild_settings(guild_id)
    try:
        channels = json.loads(settings.get("opt_out_channels") or "[]")
    except json.JSONDecodeError:
        channels = []
    channel_key = str(channel_id)
    if opt_out and channel_key not in channels:
        channels.append(channel_key)
    elif not opt_out and channel_key in channels:
        channels.remove(channel_key)
    update_guild_settings(guild_id, opt_out_channels=channels)
    return channels


def memory_user_key(user_id: int, guild_id: int | None) -> str:
    """Scope memory/relationships per server when isolation is enabled."""
    if not guild_id:
        return str(user_id)
    settings = get_guild_settings(guild_id)
    if settings.get("memory_isolated"):
        return f"{guild_id}:{user_id}"
    return str(user_id)


def get_guild_personality_mode(guild_id: int | None) -> str | None:
    if not guild_id:
        return None
    mode = get_guild_settings(guild_id).get("personality", "")
    return mode or None
