"""
Conversation State — short-term, per-user session context.

Two separate state objects per user:

  ConversationState — what's happening in the conversation
    topic, goal, activity, argument, open_thread
    These are conversation-level facts that change when the topic shifts.

  UserState — the user's current emotional/personal state
    mood, energy, stress_level
    These are user-level facts that change more slowly and decay differently.

Keeping them separate prevents a topic change from wiping the user's emotional
state, and prevents a mood shift from resetting the conversation topic.

Confidence scores:
  Each field carries a confidence float (0.0–1.0).
  Fields below CONFIDENCE_THRESHOLD are not injected into the prompt —
  a low-confidence mood guess is worse than no mood at all.

Gradual decay:
  0–15 min  → full weight (decay_factor = 1.0)
  15–30 min → partial weight (decay_factor = 0.5)
  30+ min   → expired, state is reset

Injection guard:
  State is only re-injected into the prompt when something actually changed
  (topic shift, emotional shift, goal change). Unchanged state is cached and
  reused without re-injecting, saving tokens and reducing repetition.
"""

import time
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger("corsbot.session")

# ── Thresholds ────────────────────────────────────────────────────────────── #

CONFIDENCE_THRESHOLD = 0.55   # fields below this are not injected
SESSION_TIMEOUT      = 1800   # 30 min → full expiry
SONG_TASK_TIMEOUT    = 900    # 15 min → keep song-guess state alive
DECAY_PARTIAL_AT     = 900    # 15 min → partial weight starts
REFRESH_EVERY        = 4      # re-analyze every N messages
MESSAGE_WINDOW       = 12     # messages fed to the analyzer


# ── State dataclasses ─────────────────────────────────────────────────────── #

@dataclass
class FieldWithConfidence:
    value: str = ""
    confidence: float = 0.0

    def is_usable(self) -> bool:
        return bool(self.value) and self.confidence >= CONFIDENCE_THRESHOLD

    def __str__(self) -> str:
        return self.value


@dataclass
class ConversationState:
    """What's happening in the conversation right now."""
    topic:       FieldWithConfidence = field(default_factory=FieldWithConfidence)
    goal:        FieldWithConfidence = field(default_factory=FieldWithConfidence)
    activity:    FieldWithConfidence = field(default_factory=FieldWithConfidence)
    argument:    FieldWithConfidence = field(default_factory=FieldWithConfidence)
    open_thread: FieldWithConfidence = field(default_factory=FieldWithConfidence)
    updated_at:  float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not any(f.is_usable() for f in [
            self.topic, self.goal, self.activity, self.argument, self.open_thread
        ])

    def decay_factor(self) -> float:
        age = time.time() - self.updated_at
        if age >= SESSION_TIMEOUT:
            return 0.0
        if age >= DECAY_PARTIAL_AT:
            # Linear decay from 1.0 at 15min to 0.0 at 30min — clamped to [0, 1]
            return max(0.0, 1.0 - (age - DECAY_PARTIAL_AT) / (SESSION_TIMEOUT - DECAY_PARTIAL_AT))
        return 1.0

    def to_prompt_block(self, use_priority: bool = True) -> str:
        """
        Return the conversation state as a formatted prompt block.
        
        When use_priority is True, fields are organized by priority:
        - CRITICAL: topic
        - HIGH: activity, argument, open_thread
        - MEDIUM: goal
        
        This enables intelligent trimming when token budget is tight.
        """
        decay = self.decay_factor()
        if decay == 0.0:
            return ""
        
        # Apply decay to confidence threshold — older state needs higher confidence to inject
        effective_threshold = CONFIDENCE_THRESHOLD + (1.0 - decay) * 0.2
        
        if not use_priority:
            # Legacy flat format
            lines = []
            for label, f in [
                ("topic", self.topic),
                ("goal", self.goal),
                ("activity", self.activity),
                ("ongoing argument", self.argument),
                ("open thread", self.open_thread),
            ]:
                if f.value and f.confidence >= effective_threshold:
                    lines.append(f"{label}: {f.value}")
            return "\n".join(lines)
        
        # Priority-based format
        sections = []
        
        # CRITICAL: topic (most important for context)
        if self.topic.value and self.topic.confidence >= effective_threshold:
            sections.append(f"[CRITICAL]\ntopic: {self.topic.value}")
        
        # HIGH: activity, argument, open_thread
        high_lines = []
        if self.activity.value and self.activity.confidence >= effective_threshold:
            high_lines.append(f"activity: {self.activity.value}")
        if self.argument.value and self.argument.confidence >= effective_threshold:
            high_lines.append(f"argument: {self.argument.value}")
        if self.open_thread.value and self.open_thread.confidence >= effective_threshold:
            high_lines.append(f"open_thread: {self.open_thread.value}")
        if high_lines:
            sections.append(f"[HIGH]\n" + "\n".join(high_lines))
        
        # MEDIUM: goal
        if self.goal.value and self.goal.confidence >= effective_threshold:
            sections.append(f"[MEDIUM]\ngoal: {self.goal.value}")
        
        return "\n\n".join(sections)

    def fingerprint(self) -> str:
        """Hash of the injected content — used to detect changes."""
        return hashlib.md5(self.to_prompt_block().encode()).hexdigest()[:8]


@dataclass
class UserState:
    """The user's current emotional/personal state."""
    mood:        FieldWithConfidence = field(default_factory=FieldWithConfidence)
    energy:      FieldWithConfidence = field(default_factory=FieldWithConfidence)
    stress:      FieldWithConfidence = field(default_factory=FieldWithConfidence)
    updated_at:  float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not any(f.is_usable() for f in [self.mood, self.energy, self.stress])

    def decay_factor(self) -> float:
        """User state decays faster than conversation state."""
        age = time.time() - self.updated_at
        if age >= SESSION_TIMEOUT:
            return 0.0
        if age >= DECAY_PARTIAL_AT:
            return max(0.0, 1.0 - (age - DECAY_PARTIAL_AT) / (SESSION_TIMEOUT - DECAY_PARTIAL_AT))
        return 1.0

    def to_prompt_block(self) -> str:
        decay = self.decay_factor()
        if decay == 0.0:
            return ""
        lines = []
        effective_threshold = CONFIDENCE_THRESHOLD + (1.0 - decay) * 0.2
        for label, f in [
            ("user mood", self.mood),
            ("user energy", self.energy),
            ("user stress", self.stress),
        ]:
            if f.value and f.confidence >= effective_threshold:
                lines.append(f"{label}: {f.value}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return hashlib.md5(self.to_prompt_block().encode()).hexdigest()[:8]


# ── Session store ─────────────────────────────────────────────────────────── #

@dataclass
class SongTaskState:
    active: bool = False
    clues: list[str] = field(default_factory=list)
    artist_hint: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass
class _Session:
    conv:              ConversationState = field(default_factory=ConversationState)
    user:              UserState         = field(default_factory=UserState)
    messages:          list              = field(default_factory=list)
    msg_count:         int               = 0
    last_seen:         float             = field(default_factory=time.time)
    last_conv_fp:      str               = ""   # fingerprint of last injected conv state
    last_user_fp:      str               = ""   # fingerprint of last injected user state
    last_analyzed_fp:  str               = ""   # fingerprint of messages at last analyze_state call
    song_task:         SongTaskState     = field(default_factory=SongTaskState)


_sessions: dict[int, _Session] = defaultdict(_Session)


def _is_expired(user_id: int) -> bool:
    sess = _sessions.get(user_id)
    if not sess:
        return True
    return (time.time() - sess.last_seen) > SESSION_TIMEOUT


def _song_task_is_expired(song_task: SongTaskState) -> bool:
    return not song_task.active or (time.time() - song_task.updated_at) > SONG_TASK_TIMEOUT


def _get_or_create(user_id: int) -> _Session:
    if _is_expired(user_id):
        _sessions[user_id] = _Session()
    return _sessions[user_id]


# ── Public API ────────────────────────────────────────────────────────────── #

def add_message(user_id: int, content: str):
    """Add a user message to the session message window."""
    sess = _get_or_create(user_id)
    sess.messages.append(content)
    sess.messages = sess.messages[-MESSAGE_WINDOW:]
    sess.msg_count += 1
    sess.last_seen = time.time()


def add_bot_message(user_id: int, content: str):
    """Add a bot message to the session message window for context tracking.
    
    This ensures the bot's own recent responses are available when processing
    future messages, preventing the bot from contradicting itself or appearing
    confused about what it previously said.
    """
    sess = _get_or_create(user_id)
    sess.messages.append(content)
    sess.messages = sess.messages[-MESSAGE_WINDOW:]
    sess.last_seen = time.time()
    log.debug(f"[session] added bot message user={user_id} len={len(content)}")


def should_refresh(user_id: int) -> bool:
    if _is_expired(user_id):
        return False
    sess = _sessions.get(user_id)
    if not sess:
        return False
    return sess.msg_count > 0 and sess.msg_count % REFRESH_EVERY == 0


def get_recent_messages(user_id: int) -> list[str]:
    sess = _sessions.get(user_id)
    return sess.messages if sess else []


def get_song_task_state(user_id: int) -> SongTaskState:
    sess = _sessions.get(user_id)
    if not sess:
        return SongTaskState()
    if _song_task_is_expired(sess.song_task):
        sess.song_task = SongTaskState()
        return sess.song_task
    return sess.song_task


def update_song_task_state(
    user_id: int,
    clues: list[str],
    artist_hint: str = "",
    active: bool = True,
):
    sess = _get_or_create(user_id)
    sess.song_task = SongTaskState(
        active=active,
        clues=clues[-8:],
        artist_hint=artist_hint or sess.song_task.artist_hint,
        updated_at=time.time(),
    )
    sess.last_seen = time.time()
    log.debug(
        f"[session] updated song task user={user_id} active={active} clues={len(sess.song_task.clues)} artist_hint={artist_hint!r}"
    )


def clear_song_task_state(user_id: int):
    sess = _sessions.get(user_id)
    if not sess:
        return
    sess.song_task = SongTaskState()
    sess.last_seen = time.time()
    log.debug(f"[session] cleared song task state user={user_id}")


def set_state(user_id: int, conv: ConversationState, user: UserState):
    sess = _get_or_create(user_id)
    sess.conv = conv
    sess.user = user
    sess.last_seen = time.time()
    log.debug(f"[session] updated user={user_id} conv={conv.fingerprint()} user_state={user.fingerprint()}")


def get_state_prompt(user_id: int) -> str:
    """
    Return the combined state prompt block.
    Only re-injects if the state has changed since last injection (injection guard).
    Returns empty string if both states are empty or fully decayed.
    """
    sess = _sessions.get(user_id)
    if not sess:
        return ""

    conv_block = sess.conv.to_prompt_block()
    user_block = sess.user.to_prompt_block()

    conv_fp = hashlib.md5(conv_block.encode()).hexdigest()[:8]
    user_fp = hashlib.md5(user_block.encode()).hexdigest()[:8]

    # Injection guard: if nothing changed, return cached content
    if conv_fp == sess.last_conv_fp and user_fp == sess.last_user_fp:
        log.debug(f"[session] state unchanged, skipping re-injection user={user_id}")
        # Still return the content — just don't log it as a new injection
        parts = []
        if conv_block:
            parts.append(f"Conversation context:\n{conv_block}")
        if user_block:
            parts.append(f"User state:\n{user_block}")
        return "\n\n".join(parts)

    # State changed — update fingerprints and return fresh block
    sess.last_conv_fp = conv_fp
    sess.last_user_fp = user_fp

    parts = []
    if conv_block:
        parts.append(f"Conversation context:\n{conv_block}")
    if user_block:
        parts.append(f"User state:\n{user_block}")

    result = "\n\n".join(parts)
    if result:
        log.debug(f"[session] injecting updated state user={user_id}: {result[:80]!r}")
    return result


# ── Legacy shims ──────────────────────────────────────────────────────────── #

def set_context(user_id: int, context: str):
    pass  # no-op, kept for backward compat

def get_context(user_id: int) -> str:
    return get_state_prompt(user_id)


def get_current_topic(user_id: int) -> str:
    sess = _sessions.get(user_id)
    if not sess or not sess.conv.topic.value:
        return ""
    return sess.conv.topic.value


# ── Analyzer ─────────────────────────────────────────────────────────────── #

_ANALYZER_PROMPT = """\
Analyze this conversation and extract the current state.
Reply ONLY in this exact format. For each field, also give a confidence score 0.0-1.0.
Leave value blank if unknown or uncertain (use low confidence).

topic: <value> | confidence: <0.0-1.0>
goal: <value> | confidence: <0.0-1.0>
activity: <value> | confidence: <0.0-1.0>
argument: <value> | confidence: <0.0-1.0>
open_thread: <value> | confidence: <0.0-1.0>
user_mood: <value> | confidence: <0.0-1.0>
user_energy: <high|normal|low> | confidence: <0.0-1.0>
user_stress: <high|normal|low> | confidence: <0.0-1.0>

Rules:
- topic/goal/activity are conversation-level (what's being discussed)
- user_mood/energy/stress are user-level (how the person seems to feel)
- Only give confidence > 0.7 if you're quite sure
- Be concise: 1 short phrase per value"""


def analyze_state(user_id: int) -> tuple[ConversationState, UserState]:
    """
    Run the LLM analyzer and return updated (ConversationState, UserState).
    Skips the LLM call if messages haven't changed since last analysis.
    Synchronous — run in an executor from async code.
    """
    import hashlib
    from .ai import groq_call

    messages = get_recent_messages(user_id)
    if not messages:
        sess = _sessions.get(user_id)
        if sess:
            return sess.conv, sess.user
        return ConversationState(), UserState()

    # Fingerprint the current message window
    msg_fp = hashlib.md5("\n".join(messages).encode()).hexdigest()[:8]

    sess = _sessions.get(user_id)
    if sess and sess.last_analyzed_fp == msg_fp:
        # Messages haven't changed — skip the LLM call, return cached state
        log.debug(f"[session] analyze_state skipped (no change) user={user_id}")
        return sess.conv, sess.user

    conversation = "\n".join(f"- {m}" for m in messages)
    try:
        raw, _ = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": _ANALYZER_PROMPT},
                {"role": "user",   "content": conversation},
            ],
            max_tokens=160,
            retries=2,
            timeout=10,
        )
    except Exception as e:
        log.warning(f"[session] analyze_state failed: {e}")
        if sess:
            return sess.conv, sess.user
        return ConversationState(), UserState()

    conv, user = _parse_state(raw)
    set_state(user_id, conv, user)

    # Store fingerprint so we skip next call if nothing changed
    sess = _get_or_create(user_id)
    sess.last_analyzed_fp = msg_fp

    return conv, user


def _parse_field(line: str) -> tuple[str, float]:
    """Parse 'value | confidence: 0.8' → ('value', 0.8)"""
    if "|" in line:
        value_part, conf_part = line.split("|", 1)
        value = value_part.strip()
        try:
            conf = float(conf_part.lower().replace("confidence:", "").strip())
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = 0.5
    else:
        value = line.strip()
        conf = 0.5
    return value, conf


def _parse_state(raw: str) -> tuple[ConversationState, UserState]:
    """Parse the LLM's output into ConversationState + UserState."""
    fields: dict[str, tuple[str, float]] = {}

    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value, conf = _parse_field(rest)
        if value:
            fields[key] = (value, conf)

    def fw(key: str) -> FieldWithConfidence:
        v, c = fields.get(key, ("", 0.0))
        return FieldWithConfidence(value=v, confidence=c)

    now = time.time()
    conv = ConversationState(
        topic=fw("topic"),
        goal=fw("goal"),
        activity=fw("activity"),
        argument=fw("argument"),
        open_thread=fw("open_thread"),
        updated_at=now,
    )
    user = UserState(
        mood=fw("user_mood"),
        energy=fw("user_energy"),
        stress=fw("user_stress"),
        updated_at=now,
    )
    return conv, user
