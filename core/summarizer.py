"""
Conversation Summarizer

Generates rolling summaries of conversation history to keep token usage
efficient in long conversations.

Strategy:
- Every SUMMARIZE_EVERY messages, summarize the oldest unsummarized messages
- Store the summary in the DB keyed by thread_id
- At generation time, inject: [summary] + last RECENT_MESSAGES messages
- This keeps the prompt lean regardless of conversation length

The summary is intentionally compact — it captures what was discussed,
not a transcript. The recent messages provide the actual conversational context.
"""

import time
import logging
from core.logger import get_logger

log = get_logger("corsbot.summarizer")

# How many messages to keep verbatim (always injected as-is)
RECENT_MESSAGES = 20

# Summarize when total history exceeds this
SUMMARIZE_THRESHOLD = 30

# How many messages to include in each summarization batch
SUMMARIZE_BATCH = 30

# Model for summarization — fast 8b is fine, summaries don't need 70b quality
_SUMMARY_MODEL = "llama-3.1-8b-instant"

_SUMMARY_PROMPT = """\
Summarize this Discord conversation between a user and Corsbot (a Discord bot).
Write a compact 2-4 sentence summary covering:
- What topics were discussed
- Any important facts the user shared about themselves
- The general tone/vibe of the conversation
- Any unresolved questions or ongoing threads

Write in third person. Be specific, not generic.
Example: "User discussed their Valorant ranked grind and frustration with teammates. They mentioned they're 19, from Manila, and studying CS. Conversation was casual and jokey. User asked about improving aim but the topic shifted before a full answer."

Output ONLY the summary. No labels, no preamble."""


def _count_messages(thread_id: str) -> int:
    from core.db import get_db
    _, cursor = get_db()
    cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id=?", (thread_id,))
    return cursor.fetchone()[0]


def get_summary(thread_id: str) -> str:
    """Return the stored summary for a thread, or empty string."""
    from core.db import get_db
    _, cursor = get_db()
    cursor.execute("SELECT summary FROM summaries WHERE thread_id=?", (thread_id,))
    row = cursor.fetchone()
    return row[0] if row else ""


def _store_summary(thread_id: str, summary: str, message_count: int):
    from core.db import get_db
    conn, cursor = get_db()
    cursor.execute(
        """INSERT INTO summaries (thread_id, summary, message_count, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(thread_id) DO UPDATE SET
               summary=excluded.summary,
               message_count=excluded.message_count,
               updated_at=excluded.updated_at""",
        (thread_id, summary, message_count, time.time()),
    )
    conn.commit()


def should_summarize(thread_id: str) -> bool:
    """
    Returns True if the thread has enough new messages to warrant a new summary.
    Checks if total messages > threshold AND we have more messages than last summary.
    """
    from core.db import get_db
    total = _count_messages(thread_id)
    if total < SUMMARIZE_THRESHOLD:
        return False

    _, cursor = get_db()
    cursor.execute("SELECT message_count FROM summaries WHERE thread_id=?", (thread_id,))
    row = cursor.fetchone()
    last_summarized = row[0] if row else 0

    # Summarize when we have SUMMARIZE_THRESHOLD new messages since last summary
    return (total - last_summarized) >= SUMMARIZE_THRESHOLD


def summarize_thread(thread_id: str) -> str:
    """
    Generate and store a rolling summary for a thread.
    Summarizes the oldest SUMMARIZE_BATCH messages (excluding the most recent RECENT_MESSAGES).
    Returns the summary string, or empty string on failure.
    """
    from core.db import get_db
    from core.ai import groq_call

    _, cursor = get_db()

    # Fetch all messages except the most recent RECENT_MESSAGES
    cursor.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY timestamp ASC",
        (thread_id,),
    )
    all_rows = cursor.fetchall()
    total = len(all_rows)

    if total <= RECENT_MESSAGES:
        return ""  # nothing to summarize yet

    # Summarize everything except the tail we'll keep verbatim
    to_summarize = all_rows[:total - RECENT_MESSAGES]

    # Cap batch size to avoid huge prompts
    if len(to_summarize) > SUMMARIZE_BATCH:
        to_summarize = to_summarize[-SUMMARIZE_BATCH:]

    # Build conversation text
    lines = []
    for role, content in to_summarize:
        speaker = "User" if role == "user" else "Corsbot"
        # Strip the [DisplayName]: prefix from user messages
        if content.startswith("[") and "]: " in content:
            content = content.split("]: ", 1)[1]
        lines.append(f"{speaker}: {content[:300]}")  # cap per-message length

    conversation = "\n".join(lines)

    # Check if we have an existing summary to build on
    existing = get_summary(thread_id)
    if existing:
        conversation = f"Previous summary:\n{existing}\n\nNew messages:\n{conversation}"

    try:
        summary, _ = groq_call(
            _SUMMARY_MODEL,
            [
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user", "content": conversation},
            ],
            max_tokens=150,
            retries=2,
            timeout=12,
        )
        summary = summary.strip()
        if not summary:
            return existing or ""

        _store_summary(thread_id, summary, total)
        log.info("summary_updated",
            thread_id=thread_id,
            messages_summarized=len(to_summarize),
            total_messages=total,
            summary_len=len(summary),
        )
        return summary

    except Exception as e:
        log.warning("summarize_failed", thread_id=thread_id, error=str(e))
        return existing or ""


def get_history_with_summary(thread_id: str, recent_limit: int = RECENT_MESSAGES) -> tuple[str, list]:
    """
    Returns (summary, recent_messages) for a thread.

    summary: compact text summary of older conversation (empty if not enough history)
    recent_messages: last `recent_limit` messages as dicts [{role, content}]

    Use this instead of get_history() for the generation step.
    """
    from core.db import get_db
    _, cursor = get_db()

    cursor.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY timestamp DESC LIMIT ?",
        (thread_id, recent_limit),
    )
    rows = cursor.fetchall()[::-1]
    recent = [
        {"role": r, "content": c if len(c) <= 900 else c[:899] + "…"}
        for r, c in rows
    ]

    summary = get_summary(thread_id)
    return summary, recent
