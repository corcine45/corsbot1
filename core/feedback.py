import time

from core.logger import get_logger

log = get_logger("corsbot.feedback")

# Per-user last reply state: user_id -> {reply, mood, memory_keys}
_last_reply: dict = {}


def store_last_reply(user_id: str, reply: str, mood: str, memory_keys: list):
    """Called after each reply so /rate knows what to rate."""
    _last_reply[str(user_id)] = {
        "reply": reply,
        "mood": mood,
        "memory_keys": memory_keys,
        "timestamp": time.time(),
    }


def get_last_reply(user_id: str) -> dict | None:
    entry = _last_reply.get(str(user_id))
    if not entry:
        return None
    if time.time() - entry["timestamp"] > 300:
        return None
    return entry


def store_feedback(
    user_id: str,
    thread_id: str,
    reply_snippet: str,
    rating: str,
    mood: str,
    guild_id: str | None = None,
):
    from .db import get_db

    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO feedback (user_id, thread_id, reply_snippet, rating, mood, guild_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(user_id),
            thread_id,
            reply_snippet[:200],
            rating,
            mood,
            guild_id,
            time.time(),
        ),
    )
    conn.commit()


def apply_good_rating(user_id: str, mood: str, memory_keys: list):
    from .db import get_db

    conn, cursor = get_db()
    for key in memory_keys:
        cursor.execute(
            "UPDATE memory SET reinforcement = MIN(reinforcement + 2, 20) WHERE user_id=? AND key=?",
            (str(user_id), key),
        )
    conn.commit()
    log.info("feedback_good", user_id=user_id, keys=len(memory_keys))


def apply_bad_rating(user_id: str, mood: str, memory_keys: list | None = None):
    from .db import get_db

    keys = memory_keys or []
    conn, cursor = get_db()
    for key in keys:
        cursor.execute(
            """UPDATE memory SET
                   reinforcement = MAX(reinforcement - 2, 1),
                   confidence = MAX(COALESCE(confidence, 1.0) - 0.15, 0.2)
               WHERE user_id=? AND key=?""",
            (str(user_id), key),
        )

    now = time.time()
    cursor.execute(
        """INSERT INTO memory (user_id, key, value, updated_at, memory_type, reinforcement)
           VALUES (?, 'feedback_bad_mood', ?, ?, 'temporary', 1)
           ON CONFLICT(user_id, key) DO UPDATE SET
               value=excluded.value, updated_at=excluded.updated_at""",
        (str(user_id), mood or "default", now),
    )
    conn.commit()
    log.info("feedback_bad", user_id=user_id, mood=mood, keys=len(keys))


def get_feedback_stats(user_id: str, guild_id: str | None = None) -> dict:
    from .db import get_db

    _, cursor = get_db()
    if guild_id is None:
        cursor.execute(
            "SELECT rating, COUNT(*) FROM feedback WHERE user_id=? GROUP BY rating",
            (str(user_id),),
        )
    else:
        cursor.execute(
            "SELECT rating, COUNT(*) FROM feedback WHERE user_id=? AND guild_id=? GROUP BY rating",
            (str(user_id), guild_id),
        )
    rows = cursor.fetchall()
    stats = {"good": 0, "bad": 0}
    for rating, count in rows:
        if rating in stats:
            stats[rating] = count
    return stats


def _recent_bad_mood_patterns(user_id: str, guild_id: str | None = None) -> str:
    from .db import get_db

    _, cursor = get_db()
    cutoff = time.time() - (7 * 86400)
    if guild_id is None:
        cursor.execute(
            """SELECT mood, COUNT(*) FROM feedback
               WHERE user_id=? AND rating='bad' AND timestamp > ?
               GROUP BY mood ORDER BY COUNT(*) DESC LIMIT 3""",
            (str(user_id), cutoff),
        )
    else:
        cursor.execute(
            """SELECT mood, COUNT(*) FROM feedback
               WHERE user_id=? AND guild_id=? AND rating='bad' AND timestamp > ?
               GROUP BY mood ORDER BY COUNT(*) DESC LIMIT 3""",
            (str(user_id), guild_id, cutoff),
        )
    rows = cursor.fetchall()
    if not rows:
        return ""
    moods = ", ".join(f"{mood} ({count}x)" for mood, count in rows if mood)
    return f"Recent bad ratings clustered around mood/context: {moods}."


def get_feedback_context(user_id: str, guild_id: str | None = None) -> str | None:
    """Return feedback guidance for prompt context."""
    from .db import get_db

    _, cursor = get_db()
    if guild_id is None:
        cursor.execute(
            "SELECT rating, reply_snippet, timestamp FROM feedback WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
            (str(user_id),),
        )
    else:
        cursor.execute(
            "SELECT rating, reply_snippet, timestamp FROM feedback WHERE user_id=? AND guild_id=? ORDER BY timestamp DESC LIMIT 1",
            (str(user_id), guild_id),
        )
    row = cursor.fetchone()

    parts = []
    if row:
        rating, snippet, timestamp = row
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
        note = (
            "Avoid repeating the same approach; the user rated the prior answer poorly."
            if rating == "bad"
            else "The user responded well to the prior answer; keep the same constructive style."
        )
        parts.append(f'{rating.upper()} rating on {when}: "{snippet}". {note}')

    stats = get_feedback_stats(user_id, guild_id)
    total = stats["good"] + stats["bad"]
    if total >= 3:
        bad_pct = int((stats["bad"] / total) * 100)
        if bad_pct >= 40:
            parts.append(
                f"Overall satisfaction is low ({bad_pct}% bad over {total} ratings) — be more specific, shorter, and less generic."
            )

    mood_note = _recent_bad_mood_patterns(user_id, guild_id)
    if mood_note:
        parts.append(mood_note)

    return "\n".join(parts) if parts else None
