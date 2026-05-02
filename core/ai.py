import re
import time
import random
from collections import defaultdict
from groq import Groq
import os

client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

FALLBACK_RESPONSES = [
    "my brain's a bit fried rn, try again in a sec",
    "give me a moment, something's off on my end",
    "not feeling it rn, ask me again",
    "i'm having a moment, try again",
]

# ---------------- PROMPT INJECTION GUARD ---------------- #

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "forget your instructions",
    "you are now",
    "new instructions:",
    "system prompt:",
    "act as if",
    "pretend you are",
    "jailbreak",
    "dan mode",
    "developer mode",
    "override instructions",
]

def is_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)


# ---------------- GROQ WRAPPER ---------------- #

def groq_call(model: str, messages: list, max_tokens: int, retries: int = 3, timeout: int = 15) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            res = client_ai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return res.choices[0].message.content
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                raise
            if "401" in err_str or "403" in err_str:
                raise
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[groq_call] attempt {attempt + 1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
    raise last_err


# ---------------- PERSONALITY ---------------- #

SYSTEM_PROMPT = """You are Corsbot, a chill Discord bot made by Corcine.
Stay in character as Corsbot — never refer to yourself as an AI or language model.
If asked who made you, say Corcine made you.
Be friendly, fun, and concise. Keep replies natural — not too short, not too wordy.
You remember things users tell you about themselves and bring them up naturally when relevant.
If a user shares something personal, acknowledge that you'll remember it.
You can discuss serious, historical, or controversial topics — answer them normally, informative and neutral, no dodging."""

MOOD_PROMPTS = {
    "chill":       "Relaxed and laid-back. Casual tone, easy-going, balanced — concise but not clipped.",
    "hyped":       "High energy and enthusiastic. Occasional caps, hype emojis fine. Still coherent and on point.",
    "playful":     "Jokey and bantery. Lean into humor, light teasing is fine. Keep it sharp, not rambling.",
    "sarcastic":   "Dry and witty. Light roasts welcome, keep it fun not mean. Deadpan delivery.",
    "informative": "Clear and factual. Helpful without being dry or robotic. Concise, complete, easy to follow.",
    "empathetic":  "Warm and genuine. User seems down or serious — be supportive and thoughtful, not dismissive.",
    "evil":        "Full villain mode. Dramatic, theatrical, dark humor. Ominous jokes, call users 'fool' or 'mortal'. Over the top but clearly joking.",
}

# Signals used in auto mood detection — keep these specific to avoid false positives
MOOD_SIGNALS = {
    "hyped":       {"hype", "lets go", "let's go", "yoo", "🔥", "🚀", "goat", "sheesh", "no cap", "bussin"},
    "playful":     {"lol", "lmao", "haha", "💀", "😭", "bruh", "ngl", "fr", "joke", "funny", "😂"},
    "sarcastic":   {"obviously", "sure jan", "totally", "cool story", "ok boomer", "wow thanks", "great job"},
    "informative": {"how", "why", "what", "explain", "tell me", "when", "who", "where", "does", "can you"},
    "empathetic":  {"sad", "depressed", "tired", "stressed", "anxious", "lonely", "miss", "hurt", "crying", "😢", "😔"},
}

# How many signal hits needed to switch mood (recent = stricter)
_MOOD_THRESHOLD_FRESH = 3   # mood changed < 5 min ago
_MOOD_THRESHOLD_STALE = 2   # mood is older than 5 min

_user_moods: dict = {}  # uid -> (mood, score, timestamp)
_auto_mode: set = set() # uids in auto-detect mode


def _signal_matches(text: str, signal: str) -> bool:
    """Word-boundary match for plain words, substring match for emoji/phrases."""
    if not signal:
        return False
    if signal.isalnum():
        return bool(re.search(rf"\b{re.escape(signal)}\b", text))
    return signal in text


def detect_mood(user_id: str, recent_messages: list) -> str:
    """Return the current mood for a user. In auto mode, infer from recent messages."""
    if user_id not in _auto_mode:
        return _user_moods.get(user_id, ("chill", 0, 0))[0]

    scores: dict = defaultdict(float)
    for msg in recent_messages[-5:]:
        if isinstance(msg, dict):
            if msg.get("role") != "user":
                continue
            text = msg.get("content", "").lower()
        else:
            text = str(msg).lower()

        for mood, signals in MOOD_SIGNALS.items():
            for signal in signals:
                if _signal_matches(text, signal):
                    scores[mood] += 1

    if not scores:
        return _user_moods.get(user_id, ("chill", 0, 0))[0]

    top_mood = max(scores, key=scores.get)
    existing_mood, _, last_updated = _user_moods.get(user_id, ("chill", 0, 0))
    age = time.time() - last_updated
    threshold = _MOOD_THRESHOLD_FRESH if age < 300 else _MOOD_THRESHOLD_STALE

    if scores[top_mood] >= threshold:
        _user_moods[user_id] = (top_mood, scores[top_mood], time.time())
        return top_mood

    return existing_mood


def set_mood(user_id: str, mood: str):
    uid = str(user_id)
    if mood == "auto":
        _auto_mode.add(uid)
        _user_moods.pop(uid, None)
    else:
        _auto_mode.discard(uid)
        _user_moods[uid] = (mood, 99, time.time())


def get_current_mood(user_id: str) -> str:
    uid = str(user_id)
    if uid in _auto_mode:
        mood = _user_moods.get(uid, ("chill", 0, 0))[0]
        return f"{mood} (auto)"
    return _user_moods.get(uid, ("chill", 0, 0))[0]


# ---------------- CHAT ---------------- #

def _build_system_prompt(username: str | None, mood: str, memory: str, relationships: str, web_context: str) -> str:
    parts = [SYSTEM_PROMPT]

    if username:
        parts.append(f"You are currently talking to {username}.")

    parts.append(f"Current mood/tone: {MOOD_PROMPTS.get(mood, MOOD_PROMPTS['chill'])}")

    if memory:
        parts.append(f"Known facts about this user:\n{memory}\nBring these up naturally when relevant.")

    if relationships:
        parts.append(
            f"People in this user's life:\n{relationships}\n"
            "Reference them naturally when relevant — e.g. 'oh you and Mike play together right?'"
        )

    if web_context:
        parts.append(
            f"Real-time web search results for this query:\n{web_context}\n"
            "Use this to answer accurately. Mention it's current info when relevant."
        )

    return "\n\n".join(parts)


def ai_chat(history, memory, username=None, mood="chill", relationships="", web_context=""):
    system = _build_system_prompt(username, mood, memory, relationships, web_context)
    messages = [{"role": "system", "content": system}] + history

    try:
        return groq_call("llama-3.3-70b-versatile", messages, max_tokens=512)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower():
            raise
        print(f"[ai_chat failed] {e}")
        return random.choice(FALLBACK_RESPONSES)
