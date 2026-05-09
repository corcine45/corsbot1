"""
Unified runtime user profile.

This module combines the bot's scattered user signals into one per-response
model, then compresses that model into a prompt-safe summary. The raw model is
kept structured for debugging and future policy decisions; the compressed
summary is what generation should consume.
"""

from __future__ import annotations

from typing import Any


MAX_SECTION_CHARS = 700
MAX_SUMMARY_CHARS = 2600


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _truncate(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _presence(user_status: str = "", user_activity: str = "") -> dict[str, str]:
    return {
        "status": _clean(user_status),
        "activity": _clean(user_activity),
    }


def build_unified_user_state(
    *,
    memory: str = "",
    session: str = "",
    presence: dict[str, Any] | None = None,
    emotion: dict[str, Any] | None = None,
    relationships: str = "",
    activity_patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "memory": _truncate(memory),
        "session": _truncate(session),
        "presence": presence or {},
        "emotion": emotion or {},
        "relationships": _truncate(relationships),
        "activity_patterns": activity_patterns or {},
    }


def build_user_state_from_context(ctx) -> dict[str, Any]:
    """
    Build the unified profile from an AgentContext-like object.

    Kept duck-typed so bot internals can evolve without forcing imports back
    into this low-level helper.
    """
    return build_unified_user_state(
        memory=getattr(ctx, "memory", ""),
        session=getattr(ctx, "session_context", ""),
        presence=_presence(
            getattr(ctx, "user_status", ""),
            getattr(ctx, "user_activity", ""),
        ),
        emotion={
            "raw_state": _clean(getattr(ctx, "raw_emotion_state", "")),
            "state": _clean(getattr(ctx, "emotion_state", "")),
            "momentum": _clean(getattr(ctx, "emotion_momentum", "")),
            "tone_guidance": _clean(getattr(ctx, "emotion_hint", "")),
        },
        relationships=getattr(ctx, "relationships", ""),
        activity_patterns={
            "reflection": _truncate(getattr(ctx, "reflection", "")),
            "presence": _truncate(getattr(ctx, "presence_patterns", "")),
            "feedback": _truncate(getattr(ctx, "feedback_context", ""), 400),
        },
    )


def compress_user_state(user_state: dict[str, Any], username: str = "") -> str:
    lines: list[str] = []

    memory = _clean(user_state.get("memory"))
    if memory:
        lines.append("Memory:\n" + memory)

    session = _clean(user_state.get("session"))
    if session:
        lines.append("Session:\n" + session)

    presence = user_state.get("presence") or {}
    presence_bits = []
    status = _clean(presence.get("status"))
    activity = _clean(presence.get("activity"))
    if status and status != "online":
        presence_bits.append(f"status={status}")
    if activity:
        presence_bits.append(f"activity={activity}")
    if presence_bits:
        lines.append("Presence: " + "; ".join(presence_bits))

    emotion = user_state.get("emotion") or {}
    emotion_bits = []
    raw_emotion_state = _clean(emotion.get("raw_state"))
    emotion_state = _clean(emotion.get("state"))
    momentum = _clean(emotion.get("momentum"))
    tone_guidance = _clean(emotion.get("tone_guidance"))
    if raw_emotion_state and raw_emotion_state != emotion_state:
        emotion_bits.append(f"current_signal={raw_emotion_state}")
    if emotion_state:
        emotion_bits.append(f"detected={emotion_state}")
    if momentum:
        emotion_bits.append(f"momentum={momentum}")
    if tone_guidance:
        emotion_bits.append(f"tone={tone_guidance}")
    if emotion_bits:
        lines.append("Emotion: " + "; ".join(emotion_bits))

    relationships = _clean(user_state.get("relationships"))
    if relationships:
        lines.append("Relationships:\n" + relationships)

    patterns = user_state.get("activity_patterns") or {}
    pattern_bits = []
    reflection = _clean(patterns.get("reflection"))
    presence_patterns = _clean(patterns.get("presence"))
    feedback = _clean(patterns.get("feedback"))
    if reflection:
        pattern_bits.append("behavioral insight=" + reflection)
    if presence_patterns:
        pattern_bits.append("presence patterns=" + presence_patterns)
    if feedback:
        pattern_bits.append("feedback=" + feedback)
    if pattern_bits:
        lines.append("Activity patterns:\n" + "\n".join(pattern_bits))

    if not lines:
        return ""

    subject = username or "this user"
    summary = (
        f"Unified runtime profile for {subject}. Use it silently to stay "
        "consistent; do not explicitly mention stored facts, relationships, "
        "or inferred state unless the user asks or brings them up.\n\n"
        + "\n\n".join(lines)
    )
    return _truncate(summary, MAX_SUMMARY_CHARS)
