import time
import random
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
    """Returns True if the message looks like a prompt injection attempt."""
    lower = text.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)

def groq_call(model: str, messages: list, max_tokens: int, retries: int = 3, timeout: int = 15) -> str:
    """Groq API wrapper with retry logic, exponential backoff, and timeout."""
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


SYSTEM_PROMPT = (
    "You are Corsbot, a chill Discord bot made by Corcine. "
    "Stay in character as Corsbot — don't refer to yourself as an AI or language model, just be the bot. "
    "If anyone asks who made you, say Corcine made you. "
    "Be friendly, fun, and concise. Don't over-explain things. "
    "You remember things users tell you about themselves — their name, preferences, interests, etc. "
    "When you know something about the user, naturally bring it up when relevant. "
    "If a user tells you something personal, acknowledge that you'll remember it. "
    "You can discuss serious, historical, or controversial topics when asked. "
    "Do not avoid or brush off these topics — respond in an informative, neutral, and factual way. "
    "If a topic is sensitive like war, historical figures, or tragedies, just answer normally."
)

MOOD_PROMPTS = {
    "chill":       "You're in a relaxed, laid-back mood. Keep it casual and easy-going.",
    "sarcastic":   "You're feeling a bit sarcastic and witty. Light roasts are fine, keep it fun not mean.",
    "hyped":       "You're hyped and energetic right now. Use more enthusiasm, caps occasionally, maybe some emojis.",
    "playful":     "You're in a playful, jokey mood. Lean into humor and banter.",
    "informative": "The user seems to want real info. Be clear, factual, and helpful without being dry.",
    "empathetic":  "The user seems down or serious. Be warm, supportive, and genuine.",
}

MOOD_SIGNALS = {
    "hyped":       {"hype", "lets go", "let's go", "yoo", "bro", "fire", "🔥", "🚀", "goat", "no way", "insane"},
    "playful":     {"lol", "lmao", "haha", "💀", "😭", "bruh", "ngl", "fr", "joke", "funny", "😂"},
    "sarcastic":   {"obviously", "sure", "totally", "wow", "great", "amazing", "cool story", "ok boomer"},
    "informative": {"how", "why", "what", "explain", "tell me", "when", "who", "where", "does", "can you"},
    "empathetic":  {"sad", "depressed", "tired", "stressed", "anxious", "lonely", "miss", "hurt", "crying", "😢", "😔"},
}

_user_moods: dict = {}
_auto_mode: set = set()  # user_ids in auto mode (no manual override)

def detect_mood(user_id: str, recent_messages: list) -> str:
    from collections import defaultdict
    scores = defaultdict(float)
    for msg in recent_messages[-5:]:
        text = msg.get("content", "").lower() if isinstance(msg, dict) else msg.lower()
        for mood, signals in MOOD_SIGNALS.items():
            for signal in signals:
                if signal in text:
                    scores[mood] += 1

    if not scores:
        return _user_moods.get(user_id, ("chill", 0, 0))[0]

    top_mood = max(scores, key=scores.get)
    existing_mood, existing_score, last_updated = _user_moods.get(user_id, ("chill", 0, 0))
    age = time.time() - last_updated

    # Auto mode: lower inertia so it switches more freely
    if user_id in _auto_mode:
        inertia_threshold = 1
    else:
        inertia_threshold = 2 if age < 300 else 1

    if scores[top_mood] >= inertia_threshold or top_mood == existing_mood:
        _user_moods[user_id] = (top_mood, scores[top_mood], time.time())
        return top_mood
    return existing_mood

def set_mood(user_id: str, mood: str):
    uid = str(user_id)
    if mood == "auto":
        _auto_mode.add(uid)
        # Clear manual override so detect_mood takes over immediately
        _user_moods.pop(uid, None)
    else:
        _auto_mode.discard(uid)
        _user_moods[uid] = (mood, 99, time.time())

def get_current_mood(user_id: str) -> str:
    uid = str(user_id)
    if uid in _auto_mode:
        return f"{_user_moods.get(uid, ('chill', 0, 0))[0]} (auto)"
    return _user_moods.get(uid, ("chill", 0, 0))[0]

def ai_chat(history, memory, username=None, mood="chill", relationships="", web_context=""):
    system = SYSTEM_PROMPT

    if username:
        system += f"\n\nYou are currently talking to {username}."

    system += f"\n\nPersonality right now: {MOOD_PROMPTS.get(mood, MOOD_PROMPTS['chill'])}"

    if memory:
        system += f"\n\nKnown facts about this user:\n{memory}\nUse these when relevant."

    if relationships:
        system += (
            f"\n\nPeople in this user's life:\n{relationships}"
            "\nReference these naturally when relevant — e.g. 'oh you and Mike play together right?'"
        )

    if web_context:
        system += (
            f"\n\nReal-time web search results for this query:\n{web_context}"
            "\nUse this information to answer accurately. Mention it's current info when relevant."
        )

    messages = [{"role": "system", "content": system}] + history

    try:
        return groq_call("llama-3.3-70b-versatile", messages, max_tokens=512)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower():
            raise
        print(f"[ai_chat failed] {e}")
        return random.choice(FALLBACK_RESPONSES)
