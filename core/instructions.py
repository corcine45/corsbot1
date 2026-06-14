"""
Deferred Instructions

Lets users tell the bot to do something when a condition is met.
Example: "when dimples comes online, tag him and say you're being bad"

Currently supported triggers:
- online: fires when a guild member's status changes to online/idle/dnd
"""

import logging
import re
import time

from core.db import get_db
from core.logger import get_logger

log = get_logger("corsbot.instructions")

# Patterns to detect "when X comes online" type requests
_ONLINE_PATTERNS = [
    re.compile(
        r"when\s+(.+?)\s+(?:comes?\s+online|goes?\s+online|is\s+online|gets?\s+online|logs?\s+in|comes?\s+back)",
        re.I,
    ),
    re.compile(
        r"(?:next\s+time|once)\s+(.+?)\s+(?:comes?\s+online|is\s+online|logs?\s+in)",
        re.I,
    ),
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


def store_instruction(
    requester_id: int,
    guild_id: int,
    channel_id: int,
    trigger_type: str,
    trigger_target: str,
    action: str,
    trigger_target_id: int | None = None,
):
    """Store a deferred instruction in the DB."""
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO deferred_instructions
           (requester_id, guild_id, channel_id, trigger_type, trigger_target, trigger_target_id, action, created_at, fired)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            str(requester_id),
            str(guild_id),
            str(channel_id),
            trigger_type,
            trigger_target.lower(),
            str(trigger_target_id) if trigger_target_id else None,
            action,
            time.time(),
        ),
    )
    conn.commit()
    target_label = (
        f"{trigger_target} ({trigger_target_id})"
        if trigger_target_id
        else trigger_target
    )
    log.info(
        f"[instructions] stored: when '{target_label}' {trigger_type} → {action[:60]}"
    )


def get_pending_online_instructions(
    guild_id: int, member_id: int, member_name: str
) -> list[dict]:
    """
    Fetch unfired instructions that should trigger when a member comes online.
    Prefer exact member ID matches; also allow legacy/name-based matches.
    """
    _, cursor = get_db()
    cursor.execute(
        """SELECT id, requester_id, channel_id, action
           FROM deferred_instructions
           WHERE guild_id=? AND trigger_type='online'
             AND fired=0
             AND (
                trigger_target_id=?
                OR (trigger_target_id IS NULL AND LOWER(trigger_target)=LOWER(?))
             )""",
        (str(guild_id), str(member_id), member_name.lower()),
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


def craft_online_message(
    target_name: str,
    target_id: int,
    action: str,
    requester_id: int,
) -> str:
    """
    Turn a raw deferred action into a natural Corsbot ping.
    Falls back to the raw action on failure.
    """
    from core.ai import groq_call

    mention = f"<@{target_id}>"
    fallback = f"{mention} {action}".strip()
    try:
        content, _ = groq_call(
            "llama-3.1-8b-instant",
            [
                {
                    "role": "system",
                    "content": (
                        "You are Corsbot, a chill Discord bot. "
                        "Write ONE short casual sentence to ping someone who just came online. "
                        "Sound like a friend, not a notification bot. "
                        "Must start with the target mention provided. "
                        "No quotes. No explanation. 1 sentence max."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target mention: {mention} ({target_name})\n"
                        f"Requested by user id: {requester_id}\n"
                        f"What to convey: {action}"
                    ),
                },
            ],
            max_tokens=80,
            retries=1,
            timeout=8,
        )
        text = (content or "").strip()
        if not text:
            return fallback
        if mention not in text:
            text = f"{mention} {text}"
        return text
    except Exception as e:
        log.warning("craft_online_message_failed", error=str(e))
        return fallback
