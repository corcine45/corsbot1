"""
Deferred Instructions

Lets users tell the bot to do something when a condition is met.
Example: "when dimples comes online, tag him and say you're being bad"

Currently supported triggers:
- online: fires when a guild member's status changes to online/idle/dnd
"""

import re
import time
import logging
from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.instructions")

# Patterns to detect "when X comes online" type requests
_ONLINE_PATTERNS = [
    re.compile(r"when\s+(.+?)\s+(?:comes?\s+online|goes?\s+online|is\s+online|gets?\s+online|logs?\s+in|comes?\s+back)", re.I),
    re.compile(r"(?:next\s+time|once)\s+(.+?)\s+(?:comes?\s+online|is\s+online|logs?\s+in)", re.I),
]


def parse_deferred_instruction(content: str) -> tuple[str, str] | None:
    """
    Try to parse a deferred instruction from a message.
    Returns (trigger_target, action) or None if not a deferred instruction.

    Example: "when dimples comes online tell him he's being bad"
    → trigger_target="dimples", action="tell him he's being bad"
    """
    for pattern in _ONLINE_PATTERNS:
        m = pattern.search(content.lower())
        if m:
            trigger_target = m.group(1).strip()
            # Extract the action — everything after the trigger phrase
            end = m.end()
            action = content[end:].strip().lstrip(",").strip()
            if not action:
                action = "tag them"
            return trigger_target, action
    return None


def store_instruction(requester_id: int, guild_id: int, channel_id: int,
                      trigger_type: str, trigger_target: str, action: str):
    """Store a deferred instruction in the DB."""
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO deferred_instructions
           (requester_id, guild_id, channel_id, trigger_type, trigger_target, action, created_at, fired)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
        (str(requester_id), str(guild_id), str(channel_id),
         trigger_type, trigger_target.lower(), action, time.time()),
    )
    conn.commit()
    log.info(f"[instructions] stored: when '{trigger_target}' {trigger_type} → {action[:60]}")


def get_pending_online_instructions(guild_id: int, member_name: str) -> list[dict]:
    """
    Fetch unfired instructions that should trigger when member_name comes online.
    Matches by display name (case-insensitive).
    """
    _, cursor = get_db()
    cursor.execute(
        """SELECT id, requester_id, channel_id, action
           FROM deferred_instructions
           WHERE guild_id=? AND trigger_type='online'
             AND LOWER(trigger_target)=LOWER(?)
             AND fired=0""",
        (str(guild_id), member_name.lower()),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "requester_id": r[1], "channel_id": r[2], "action": r[3]}
        for r in rows
    ]


def mark_instruction_fired(instruction_id: int):
    """Mark an instruction as fired so it doesn't repeat."""
    conn, cursor = get_db()
    cursor.execute(
        "UPDATE deferred_instructions SET fired=1 WHERE id=?",
        (instruction_id,),
    )
    conn.commit()
