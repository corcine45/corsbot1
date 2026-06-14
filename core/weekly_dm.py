"""
Weekly memory digest DMs — a casual recap of what Corsbot remembers.
"""

import time

import discord

from core.db import get_db
from core.logger import get_logger
from core.memory import get_memory, get_reflection

log = get_logger("corsbot.weekly_dm")

WEEK_SECONDS = 7 * 86400
OPT_OUT_KEY = "weekly_dm"


def is_opted_out(user_id: int) -> bool:
    _, cursor = get_db()
    cursor.execute(
        "SELECT value FROM memory WHERE user_id=? AND key=?",
        (str(user_id), OPT_OUT_KEY),
    )
    row = cursor.fetchone()
    return bool(row and str(row[0]).lower() in ("off", "false", "no", "0"))


def set_weekly_dm_preference(user_id: int, enabled: bool):
    conn, cursor = get_db()
    if enabled:
        cursor.execute(
            "DELETE FROM memory WHERE user_id=? AND key=?",
            (str(user_id), OPT_OUT_KEY),
        )
    else:
        cursor.execute(
            """INSERT INTO memory (user_id, key, value, updated_at, memory_type)
               VALUES (?, ?, 'off', ?, 'preference')
               ON CONFLICT(user_id, key) DO UPDATE SET value='off', updated_at=?""",
            (str(user_id), OPT_OUT_KEY, time.time(), time.time()),
        )
    conn.commit()


def _last_digest_at(user_id: int) -> float:
    _, cursor = get_db()
    cursor.execute(
        "SELECT last_digest_at FROM user_settings WHERE user_id=?",
        (str(user_id),),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else 0.0


def _mark_digest_sent(user_id: int):
    conn, cursor = get_db()
    now = time.time()
    cursor.execute(
        """INSERT INTO user_settings (user_id, last_digest_at)
           VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET last_digest_at=?""",
        (str(user_id), now, now),
    )
    conn.commit()


def get_eligible_users() -> list[int]:
    """Users with memory who chatted recently and haven't had a digest this week."""
    _, cursor = get_db()
    cutoff = time.time() - (30 * 86400)
    cursor.execute(
        """SELECT DISTINCT m.user_id
           FROM memory m
           WHERE m.key != ?
             AND EXISTS (
                 SELECT 1 FROM messages msg
                 WHERE msg.thread_id LIKE '%user:' || m.user_id
                   AND msg.timestamp >= ?
             )""",
        (OPT_OUT_KEY, cutoff),
    )
    candidates = [int(row[0]) for row in cursor.fetchall()]
    now = time.time()
    return [
        uid
        for uid in candidates
        if not is_opted_out(uid) and (now - _last_digest_at(uid)) >= WEEK_SECONDS
    ]


def build_digest_message(user_id: int) -> str:
    facts = get_memory(user_id, top_k=8)
    reflection = get_reflection(str(user_id))
    lines = [
        "yo — weekly brain dump 🧠",
        "here's what i've got on you rn:",
    ]
    if facts:
        for line in facts.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"• {line}")
    else:
        lines.append("• not much yet — talk to me more lol")
    if reflection:
        lines.append(f"\n_vibe check_: {reflection}")
    lines.append("\nreply anytime. `/memory-digest off` to stop these.")
    return "\n".join(lines)


async def send_weekly_digests(client: discord.Client):
    """Send weekly memory digests to eligible users."""
    sent = 0
    for user_id in get_eligible_users():
        try:
            user = await client.fetch_user(user_id)
            if not user:
                continue
            message = build_digest_message(user_id)
            await user.send(message[:1900])
            _mark_digest_sent(user_id)
            sent += 1
            log.info("weekly_digest_sent", user_id=user_id)
        except discord.Forbidden:
            log.debug("weekly_digest_blocked", user_id=user_id)
        except Exception as e:
            log.warning("weekly_digest_failed", user_id=user_id, error=str(e))
    if sent:
        log.info("weekly_digest_batch", count=sent)
