"""
Agent trace storage for admin observability.
"""

import time

from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.traces")


def store_agent_trace(
    user_id: int,
    guild_id: int | None,
    trace_summary: str,
    latency_ms: float,
    route: str = "",
):
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO agent_traces
           (user_id, guild_id, trace_summary, latency_ms, route, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            str(user_id),
            str(guild_id) if guild_id else None,
            trace_summary[:2000],
            round(latency_ms),
            route,
            time.time(),
        ),
    )
    conn.commit()
    # Keep table lean — prune old traces
    cursor.execute(
        """DELETE FROM agent_traces WHERE id NOT IN (
               SELECT id FROM agent_traces ORDER BY created_at DESC LIMIT 200
           )"""
    )
    conn.commit()


def get_last_guild_trace(guild_id: int) -> dict | None:
    _, cursor = get_db()
    cursor.execute(
        """SELECT user_id, trace_summary, latency_ms, route, created_at
           FROM agent_traces WHERE guild_id=? ORDER BY created_at DESC LIMIT 1""",
        (str(guild_id),),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "trace_summary": row[1],
        "latency_ms": row[2],
        "route": row[3],
        "created_at": row[4],
    }


def get_last_user_trace(user_id: int) -> dict | None:
    _, cursor = get_db()
    cursor.execute(
        """SELECT trace_summary, latency_ms, route, created_at
           FROM agent_traces WHERE user_id=? ORDER BY created_at DESC LIMIT 1""",
        (str(user_id),),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "trace_summary": row[0],
        "latency_ms": row[1],
        "route": row[2],
        "created_at": row[3],
    }


def get_recent_guild_traces(guild_id: int, limit: int = 3) -> list[dict]:
    _, cursor = get_db()
    cursor.execute(
        """SELECT user_id, trace_summary, latency_ms, route, created_at
           FROM agent_traces WHERE guild_id=? ORDER BY created_at DESC LIMIT ?""",
        (str(guild_id), limit),
    )
    return [
        {
            "user_id": row[0],
            "trace_summary": row[1],
            "latency_ms": row[2],
            "route": row[3],
            "created_at": row[4],
        }
        for row in cursor.fetchall()
    ]
