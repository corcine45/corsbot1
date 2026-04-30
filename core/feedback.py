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

def store_feedback(user_id: str, thread_id: str, reply_snippet: str, rating: str, mood: str):
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO feedback (user_id, thread_id, reply_snippet, rating, mood, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (str(user_id), thread_id, reply_snippet[:200], rating, mood, time.time()),
    )
    conn.commit()

def apply_good_rating(user_id: str, mood: str, memory_keys: list):
    """
    Good rating:
    - Boost reinforcement of memories that were active during this reply
    - Lock in the current mood for longer (increase inertia)
    """
    from .ai import _user_moods
    conn, cursor = get_db()

    # Boost reinforcement on active memory keys
    for key in memory_keys:
        cursor.execute(
            "UPDATE memory SET reinforcement = MIN(reinforcement + 2, 20) WHERE user_id=? AND key=?",
            (str(user_id), key)
        )

    conn.commit()

    # Strengthen mood inertia by resetting its timestamp to now with a high score
    existing = _user_moods.get(str(user_id), (mood, 1, 0))
    _user_moods[str(user_id)] = (existing[0], existing[1] + 3, time.time())
    print(f"[feedback] good — boosted {len(memory_keys)} memory keys, reinforced mood '{mood}'")

def apply_bad_rating(user_id: str, mood: str):
    """
    Bad rating:
    - Decay the current mood score so it's more likely to switch next message
    - Don't touch memory — we don't know what was wrong
    """
    from .ai import _user_moods
    existing = _user_moods.get(str(user_id), (mood, 1, 0))
    new_score = max(existing[1] - 2, 0)
    _user_moods[str(user_id)] = (existing[0], new_score, existing[2])
    print(f"[feedback] bad — decayed mood '{mood}' score to {new_score}")

def get_feedback_stats(user_id: str) -> dict:
    """Return good/bad counts for a user."""
    _, cursor = get_db()
    cursor.execute(
        "SELECT rating, COUNT(*) FROM feedback WHERE user_id=? GROUP BY rating",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    stats = {"good": 0, "bad": 0}
    for rating, count in rows:
        if rating in stats:
            stats[rating] = count
    return stats
