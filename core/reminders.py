"""
Scheduled reminders — ping users after a delay.

Supports natural language ("remind me in 30 minutes to ...") and /remind slash command.
"""

import logging
import re
import time

from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.reminders")

_TIME_MULTIPLIERS = {
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "day": 86400,
    "days": 86400,
}

_REMINDER_PATTERNS = [
    re.compile(
        r"remind me in (\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\s+(?:to\s+)?(.+)",
        re.I,
    ),
    re.compile(
        r"ping me in (\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\s+(?:about\s+|to\s+)?(.+)",
        re.I,
    ),
    re.compile(
        r"in (\d+)\s*(minutes?|mins?|hours?|hrs?|days?)\s+remind me\s+(?:to\s+)?(.+)",
        re.I,
    ),
]

MAX_DELAY_SECONDS = 7 * 86400  # 7 days


def _delay_seconds(amount: int, unit: str) -> int:
    unit = unit.lower().rstrip("s") if unit.lower() not in _TIME_MULTIPLIERS else unit.lower()
    if unit.endswith("s") and unit not in _TIME_MULTIPLIERS:
        unit = unit[:-1]
    multiplier = _TIME_MULTIPLIERS.get(unit, _TIME_MULTIPLIERS.get(unit + "s", 60))
    return amount * multiplier


def parse_reminder(content: str) -> tuple[float, str] | None:
    """
    Parse a reminder from natural language.
    Returns (fire_at_unix, message) or None.
    """
    text = content.strip()
    for pattern in _REMINDER_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        amount = int(match.group(1))
        unit = match.group(2).lower()
        message = match.group(3).strip().rstrip("!?.")
        if not message:
            continue
        delay = _delay_seconds(amount, unit)
        if delay <= 0 or delay > MAX_DELAY_SECONDS:
            continue
        return time.time() + delay, message
    return None


def store_reminder(
    user_id: int,
    guild_id: int | None,
    channel_id: int,
    message: str,
    fire_at: float,
) -> int:
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO reminders (user_id, guild_id, channel_id, message, fire_at, created_at, fired)
           VALUES (?, ?, ?, ?, ?, ?, 0)""",
        (
            str(user_id),
            str(guild_id) if guild_id else None,
            str(channel_id),
            message[:500],
            fire_at,
            time.time(),
        ),
    )
    conn.commit()
    reminder_id = cursor.lastrowid
    log.info(
        "reminder_stored",
        user_id=user_id,
        fire_at=round(fire_at),
        message=message[:60],
    )
    return reminder_id


def get_due_reminders() -> list[dict]:
    _, cursor = get_db()
    now = time.time()
    cursor.execute(
        """SELECT id, user_id, channel_id, message
           FROM reminders WHERE fired=0 AND fire_at<=? ORDER BY fire_at ASC LIMIT 20""",
        (now,),
    )
    return [
        {"id": r[0], "user_id": r[1], "channel_id": r[2], "message": r[3]}
        for r in cursor.fetchall()
    ]


def mark_reminder_fired(reminder_id: int):
    conn, cursor = get_db()
    cursor.execute("UPDATE reminders SET fired=1 WHERE id=?", (reminder_id,))
    conn.commit()


def get_user_pending_reminders(user_id: int) -> list[dict]:
    _, cursor = get_db()
    cursor.execute(
        """SELECT id, message, fire_at, channel_id
           FROM reminders WHERE user_id=? AND fired=0 ORDER BY fire_at ASC LIMIT 10""",
        (str(user_id),),
    )
    return [
        {"id": r[0], "message": r[1], "fire_at": r[2], "channel_id": r[3]}
        for r in cursor.fetchall()
    ]


def cancel_reminder(reminder_id: int, user_id: int) -> bool:
    conn, cursor = get_db()
    cursor.execute(
        "UPDATE reminders SET fired=1 WHERE id=? AND user_id=? AND fired=0",
        (reminder_id, str(user_id)),
    )
    conn.commit()
    return cursor.rowcount > 0


def count_pending_reminders(guild_id: int | None = None) -> int:
    _, cursor = get_db()
    if guild_id:
        cursor.execute(
            "SELECT COUNT(*) FROM reminders WHERE guild_id=? AND fired=0",
            (str(guild_id),),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM reminders WHERE fired=0")
    return cursor.fetchone()[0]
