import time

from .db import get_db

MIN_PATTERN_COUNT = 2
MAX_SUMMARY_PATTERNS = 4


def _clean_subject(subject: str) -> str:
    subject = (subject or "").strip()
    if not subject:
        return ""
    # Keep Spotify song details to show what the user is listening to
    return subject[:80]


def _is_night(ts: float | None = None) -> bool:
    hour = time.localtime(ts or time.time()).tm_hour
    return hour >= 20 or hour < 5


def _summary(kind: str, subject: str, count: int, night_count: int, social_count: int) -> str:
    frequency = "frequently" if count >= 3 else "sometimes"
    when = " at night" if night_count >= max(2, count // 2) else ""
    social = " with friends" if social_count >= max(2, count // 2) else ""

    if kind == "activity":
        if subject == "Spotify":
            return f"user {frequency} listens to Spotify{when}{social}"
        return f"user {frequency} plays {subject}{when}{social}"
    if kind == "voice":
        return f"user {frequency} joins voice chat{when}{social}"
    if kind == "status":
        return f"user is {frequency} {subject}{when}"
    return f"user {frequency} has presence pattern: {subject}{when}{social}"


def record_presence_pattern(
    user_id,
    kind: str,
    subject: str,
    *,
    is_social: bool = False,
    ts: float | None = None,
):
    """
    Store only compressed presence aggregates.

    This intentionally does not create raw event rows like "played Valorant".
    Repeated events update a single pattern row and summary.
    """
    subject = _clean_subject(subject)
    if not subject:
        return

    now = ts or time.time()
    kind = (kind or "activity").strip().lower()
    pattern_key = f"{kind}:{subject.lower()}"
    night_inc = 1 if _is_night(now) else 0
    social_inc = 1 if is_social else 0

    conn, cursor = get_db()
    cursor.execute(
        "SELECT count, night_count, social_count FROM presence_patterns WHERE user_id=? AND pattern_key=?",
        (str(user_id), pattern_key),
    )
    row = cursor.fetchone()
    if row:
        count, night_count, social_count = row
        count += 1
        night_count += night_inc
        social_count += social_inc
    else:
        count, night_count, social_count = 1, night_inc, social_inc

    summary = _summary(kind, subject, count, night_count, social_count)
    cursor.execute(
        """INSERT INTO presence_patterns
           (user_id, pattern_key, kind, subject, count, night_count, social_count, updated_at, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id, pattern_key) DO UPDATE SET
               count=excluded.count,
               night_count=excluded.night_count,
               social_count=excluded.social_count,
               updated_at=excluded.updated_at,
               summary=excluded.summary""",
        (str(user_id), pattern_key, kind, subject, count, night_count, social_count, now, summary),
    )
    conn.commit()


def get_presence_patterns(user_id, limit: int = MAX_SUMMARY_PATTERNS) -> str:
    _, cursor = get_db()
    cursor.execute(
        """SELECT summary FROM presence_patterns
           WHERE user_id=? AND count >= ?
           ORDER BY count DESC, updated_at DESC
           LIMIT ?""",
        (str(user_id), MIN_PATTERN_COUNT, limit),
    )
    return "\n".join(row[0] for row in cursor.fetchall() if row[0])


def describe_activity(activity) -> str:
    if not activity:
        return ""
    name = getattr(activity, "name", "") or ""
    details = getattr(activity, "details", "") or ""
    # For Spotify, include song/artist details if available
    if name.lower() == "spotify" and details:
        return f"{name}: {details}"
    return name


# ── Contextual Proactivity System ──────────────────────────────────────────── #
#
# Detects behavioral patterns and triggers subtle, natural bot responses.
# Key principle: Subtle > explicit. Never say "I noticed your pattern changed."
#
# Trigger patterns:
# - joins VC alone (after pattern of joining with others)
# - late-night sadness patterns (night activity + depressed keywords)
# - repeated rage gaming (frustrating game + angry patterns)
# - long silence after emotional convo (gap after deep talk)
# - sudden activity change (stop/start pattern break)

import random

# Proactivity trigger definitions
_PROACTIVITY_TRIGGERS = {
    "solo_vc": {
        "condition": "User joins voice channel alone after usually joining with friends",
        "responses": [
            "quiet tonight?",
            "where's the squad?",
            "solo vc session?",
            "missing the usual crew?",
        ],
        "cooldown_hours": 6,
    },
    "late_night_vibes": {
        "condition": "User active late at night with sad/melancholy patterns",
        "responses": [
            "rough night?",
            "can't sleep?",
            "late night thoughts hitting different huh",
            "you good?",
        ],
        "cooldown_hours": 12,
    },
    "rage_game": {
        "condition": "User repeatedly playing frustrating/competitive game",
        "responses": [
            "game being toxic again?",
            "need a break from that mess?",
            "that game testing your patience huh",
            "lmao you're still playing that?",
        ],
        "cooldown_hours": 3,
    },
    "post_emotional_silence": {
        "condition": "Long silence after an emotional/deep conversation",
        "responses": [
            "hey, you around?",
            "still thinking about earlier?",
            "hope you're doing okay",
            "checking in",
        ],
        "cooldown_hours": 2,
    },
    "sudden_stop": {
        "condition": "User suddenly stops a frequent activity pattern",
        "responses": [
            "taking a break from that?",
            "moved on already?",
            "that was quick lmao",
            "done with that phase?",
        ],
        "cooldown_hours": 6,
    },
    "returning_after_absence": {
        "condition": "User returns after long absence",
        "responses": [
            "look who's back",
            "been a minute",
            "alive huh",
            "where you been",
        ],
        "cooldown_hours": 24,
    },
}

# Track last proactivity time per user per trigger
_proactivity_cooldowns: dict[str, dict[str, float]] = {}

# Track conversation context for post-emotional silence detection
_emotional_convo_end: dict[str, float] = {}  # user_id -> timestamp of last emotional convo end


def should_be_proactive(user_id: str, context: dict) -> str | None:
    """
    Check if we should send a proactive message based on current context.
    
    Context should include:
    - activity: current activity (game, spotify, etc.)
    - voice_status: "alone", "with_friends", None
    - emotion: current emotion state
    - time_hour: current hour (0-23)
    - last_message_age: seconds since last user message
    - presence_patterns: dict of pattern_key -> pattern info
    
    Returns a subtle proactive message string, or None if no trigger.
    """
    now = time.time()
    uid = str(user_id)
    
    # Check cooldowns first
    user_cooldowns = _proactivity_cooldowns.get(uid, {})
    
    # Check each trigger
    for trigger_name, trigger_info in _PROACTIVITY_TRIGGERS.items():
        # Skip if on cooldown
        if user_cooldowns.get(trigger_name, 0) + (trigger_info["cooldown_hours"] * 3600) > now:
            continue
        
        # Check trigger conditions
        if _check_trigger(trigger_name, uid, context):
            # Set cooldown
            user_cooldowns[trigger_name] = now
            _proactivity_cooldowns[uid] = user_cooldowns
            
            # Pick random response
            return random.choice(trigger_info["responses"])
    
    return None


def _check_trigger(trigger_name: str, user_id: str, context: dict) -> bool:
    """Check if a specific trigger condition is met."""
    
    if trigger_name == "solo_vc":
        return _check_solo_vc(user_id, context)
    elif trigger_name == "late_night_vibes":
        return _check_late_night_vibes(user_id, context)
    elif trigger_name == "rage_game":
        return _check_rage_game(user_id, context)
    elif trigger_name == "post_emotional_silence":
        return _check_post_emotional_silence(user_id, context)
    elif trigger_name == "sudden_stop":
        return _check_sudden_stop(user_id, context)
    elif trigger_name == "returning_after_absence":
        return _check_returning_after_absence(user_id, context)
    
    return False


def _check_solo_vc(user_id: str, context: dict) -> bool:
    """Check if user joined VC alone after usually being social."""
    voice_status = context.get("voice_status")
    if voice_status != "alone":
        return False
    
    # Check if they usually join with friends
    patterns = context.get("presence_patterns", {})
    vc_pattern = patterns.get("voice", {})
    
    # If they frequently join VC with friends, solo is notable
    social_vc_count = vc_pattern.get("social_count", 0)
    total_vc_count = vc_pattern.get("count", 0)
    
    return social_vc_count >= 2 and total_vc_count >= 3


def _check_late_night_vibes(user_id: str, context: dict) -> bool:
    """Check for late-night activity with sad/melancholy patterns."""
    time_hour = context.get("time_hour", 12)
    emotion = context.get("emotion", "")
    
    # Must be late night (after 11pm or before 5am)
    if not (time_hour >= 23 or time_hour < 5):
        return False
    
    # Check for sad/depressed emotion
    sad_emotions = {"depressed", "lonely", "anxious", "sad", "melancholy"}
    if emotion in sad_emotions:
        return True
    
    # Check for late-night activity pattern
    patterns = context.get("presence_patterns", {})
    for pattern_info in patterns.values():
        if pattern_info.get("night_count", 0) >= 3:
            if emotion in {"sad", "lonely", "depressed", "anxious", "melancholy"}:
                return True
    
    return False


def _check_rage_game(user_id: str, context: dict) -> bool:
    """Check for repeated frustrating game sessions."""
    activity = context.get("activity", "")
    emotion = context.get("emotion", "")
    
    # Competitive/frustrating games
    rage_games = {"valorant", "league", "league of legends", "overwatch", "cs2", "counter-strike",
                  "dota", "dota 2", "apex", "apex legends", "ranked", "competitive"}
    
    if not activity or activity.lower() not in rage_games:
        return False
    
    # Check if they've been playing this game repeatedly
    patterns = context.get("presence_patterns", {})
    game_pattern = patterns.get(f"activity:{activity.lower()}", {})
    game_count = game_pattern.get("count", 0)
    
    # Trigger if they've played this game 3+ times AND showing frustration
    frustrated_emotions = {"frustrated", "angry", "annoyed", "tilted", "rage"}
    
    return game_count >= 3 and (emotion in frustrated_emotions or game_count >= 5)


def _check_post_emotional_silence(user_id: str, context: dict) -> bool:
    """Check for long silence after an emotional conversation."""
    last_emotional = _emotional_convo_end.get(str(user_id), 0)
    if last_emotional == 0:
        return False
    
    last_msg_age = context.get("last_message_age", 999999)
    
    # Check if it's been 5-30 minutes since emotional convo ended
    # (long enough to notice silence, not so long it's weird)
    return 300 <= last_msg_age <= 1800  # 5-30 minutes


def _check_sudden_stop(user_id: str, context: dict) -> bool:
    """Check if user suddenly stopped a frequent activity."""
    # This would need activity history tracking
    # For now, check if they had a strong pattern that's now absent
    patterns = context.get("presence_patterns", {})
    current_activity = context.get("activity", "")
    
    for key, pattern_info in patterns.items():
        if key.startswith("activity:"):
            activity_name = key.split(":", 1)[1]
            count = pattern_info.get("count", 0)
            updated = pattern_info.get("updated_at", 0)
            
            # If they had a strong pattern (5+ times) but it's been a while
            if count >= 5 and (time.time() - updated) > 3600:  # 1+ hour ago
                if not current_activity or current_activity.lower() != activity_name:
                    return True
    
    return False


def _check_returning_after_absence(user_id: str, context: dict) -> bool:
    """Check if user returns after long absence."""
    last_msg_age = context.get("last_message_age", 0)
    
    # If it's been more than 24 hours since last message
    return last_msg_age >= 86400  # 24 hours


def record_emotional_conversation_end(user_id: str):
    """Record that an emotional conversation just ended (for post-emotional silence detection)."""
    _emotional_convo_end[str(user_id)] = time.time()


def get_proactive_context(user_id: str) -> dict:
    """
    Build the context dict needed for proactivity checks.
    Call this from the main bot loop when processing messages.
    """
    _, cursor = get_db()
    
    # Get presence patterns
    cursor.execute(
        "SELECT kind, subject, count, night_count, social_count, updated_at FROM presence_patterns "
        "WHERE user_id=? AND count >= ?",
        (str(user_id), MIN_PATTERN_COUNT)
    )
    
    patterns = {}
    for kind, subject, count, night_count, social_count, updated_at in cursor.fetchall():
        key = f"{kind}:{subject.lower()}"
        patterns[key] = {
            "kind": kind,
            "subject": subject,
            "count": count,
            "night_count": night_count,
            "social_count": social_count,
            "updated_at": updated_at,
        }
    
    # Group by kind for easier access
    by_kind = {}
    for key, info in patterns.items():
        kind = info["kind"]
        if kind not in by_kind:
            by_kind[kind] = info
        elif info["count"] > by_kind[kind].get("count", 0):
            by_kind[kind] = info
    
    return {
        "presence_patterns": by_kind,
        "all_patterns": patterns,
    }
