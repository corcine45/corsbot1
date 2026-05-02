import time
from .db import get_db

# Per-user last reply state: user_id -> {reply, mood, memory_keys}
_last_reply: dict = {}

def store_last_reply(user_id: str, reply: str, mood: str, memory_keys: list):
    """Called after each reply so !rate knows what to rate."""
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
    # Expire after 5 minutes — rating a reply from an hour ago makes no sense
    if time.time() - entry["timestamp"] > 300:
        return None
    return entry

def store_feedback(user_id: str, thread_id: str, reply_snippet: str, rating: str, mood: str, guild_id: str | None = None):
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO feedback (user_id, thread_id, reply_snippet, rating, mood, guild_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(user_id), thread_id, reply_snippet[:200], rating, mood, guild_id, time.time()),
    )
    conn.commit()

def apply_good_rating(user_id: str, mood: str, memory_keys: list):
    conn, cursor = get_db()
    for key in memory_keys:
        cursor.execute(
            "UPDATE memory SET reinforcement = MIN(reinforcement + 2, 20) WHERE user_id=? AND key=?",
            (str(user_id), key)
        )
    conn.commit()
    print(f"[feedback] good — boosted {len(memory_keys)} memory keys")

def apply_bad_rating(user_id: str, mood: str):
    print(f"[feedback] bad rating noted")

def get_feedback_stats(user_id: str, guild_id: str | None = None) -> dict:
    """Return good/bad counts for a user, optionally filtered by server."""
    _, cursor = get_db()
    if guild_id is None:
        cursor.execute(
            "SELECT rating, COUNT(*) FROM feedback WHERE user_id=? GROUP BY rating",
            (str(user_id),)
        )
    else:
        cursor.execute(
            "SELECT rating, COUNT(*) FROM feedback WHERE user_id=? AND guild_id=? GROUP BY rating",
            (str(user_id), guild_id)
        )
    rows = cursor.fetchall()
    stats = {"good": 0, "bad": 0}
    for rating, count in rows:
        if rating in stats:
            stats[rating] = count
    return stats


def get_feedback_context(user_id: str, guild_id: str | None = None) -> str | None:
    """Return the most recent feedback summary for prompt context."""
    _, cursor = get_db()
    if guild_id is None:
        cursor.execute(
            "SELECT rating, reply_snippet, timestamp FROM feedback WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
            (str(user_id),)
        )
    else:
        cursor.execute(
            "SELECT rating, reply_snippet, timestamp FROM feedback WHERE user_id=? AND guild_id=? ORDER BY timestamp DESC LIMIT 1",
            (str(user_id), guild_id)
        )
    row = cursor.fetchone()
    if not row:
        return None
    rating, snippet, timestamp = row
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
    note = (
        "Avoid repeating the same approach; the user rated the prior answer poorly."
        if rating == "bad" else
        "The user responded well to the prior answer; keep the same constructive style."
    )
    return f"{rating.upper()} rating on {when}: \"{snippet}\". {note}"
