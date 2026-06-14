"""
Persist music queues across restarts.
Stream URLs expire — we store metadata and re-resolve on playback.
"""

import json
import time

from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.music_store")


def _slim_track(track: dict | None) -> dict | None:
    if not track:
        return None
    return {
        "title": track.get("title") or "unknown track",
        "webpage_url": track.get("webpage_url") or "",
        "search_query": track.get("search_query")
        or track.get("webpage_url")
        or track.get("title")
        or "",
        "duration": track.get("duration"),
        "requested_by": track.get("requested_by"),
    }


def save_guild_music(guild_id: int, queue: list[dict], now_playing: dict | None):
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO music_queues (guild_id, now_playing, queue, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(guild_id) DO UPDATE SET
               now_playing=excluded.now_playing,
               queue=excluded.queue,
               updated_at=excluded.updated_at""",
        (
            str(guild_id),
            json.dumps(_slim_track(now_playing)),
            json.dumps([_slim_track(t) for t in queue if t]),
            time.time(),
        ),
    )
    conn.commit()


def delete_guild_music(guild_id: int):
    conn, cursor = get_db()
    cursor.execute("DELETE FROM music_queues WHERE guild_id=?", (str(guild_id),))
    conn.commit()


def load_all_music() -> dict[int, dict]:
    _, cursor = get_db()
    cursor.execute("SELECT guild_id, now_playing, queue FROM music_queues")
    result = {}
    for guild_id, now_raw, queue_raw in cursor.fetchall():
        try:
            now_playing = json.loads(now_raw) if now_raw else None
            queue = json.loads(queue_raw) if queue_raw else []
        except json.JSONDecodeError:
            continue
        result[int(guild_id)] = {
            "now_playing": now_playing,
            "queue": [t for t in queue if t],
        }
    return result
