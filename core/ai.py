import time
import random
import os
from google import genai
from google.genai import types

client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHAT_MODEL = "gemini-2.0-flash"
FAST_MODEL = "gemini-2.0-flash-lite"

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

# ---------------- GEMINI CALL ---------------- #

def gemini_call(system: str, history: list, user_message: str,
                model: str = CHAT_MODEL, max_tokens: int = 512,
                retries: int = 3) -> str:
    """Call Gemini with retry logic and exponential backoff."""
    # Convert OpenAI-style history to Gemini format
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    last_err = None
    for attempt in range(retries):
        try:
            chat = client_ai.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
                history=gemini_history,
            )
            res = chat.send_message(user_message)
            return res.text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                raise
            if "401" in err_str or "403" in err_str or "API_KEY" in err_str:
                raise
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[gemini_call] attempt {attempt + 1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
    raise last_err

def gemini_simple(system: str, prompt: str, model: str = FAST_MODEL,
                  max_tokens: int = 100, retries: int = 2) -> str:
    """Single-turn call for lightweight tasks."""
    last_err = None
    for attempt in range(retries):
        try:
            res = client_ai.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
            return res.text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                raise
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_err

def groq_call(model: str, messages: list, max_tokens: int,
              retries: int = 2, timeout: int = 10) -> str:
    """Compatibility shim — routes lightweight calls to gemini_simple."""
    system = ""
    user_msg = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user_msg = m["content"]
    return gemini_simple(system, user_msg, model=FAST_MODEL,
                         max_tokens=max_tokens, retries=retries)

# ---------------- PERSONALITY ---------------- #

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
    "evil":        "You're in full villain mode. Be dramatic, menacing, and theatrical. Talk like a supervillain — dark humor, ominous threats that are clearly jokes, call users 'fool' or 'mortal'. Keep it fun and over the top, not actually mean.",
}

MOOD_SIGNALS = {
    "hyped":       {"hype", "lets go", "let's go", "yoo", "bro", "fire", "🔥", "🚀", "goat", "no way", "insane"},
    "playful":     {"lol", "lmao", "haha", "💀", "😭", "bruh", "ngl", "fr", "joke", "funny", "😂"},
    "sarcastic":   {"obviously", "sure", "totally", "wow", "great", "amazing", "cool story", "ok boomer"},
    "informative": {"how", "why", "what", "explain", "tell me", "when", "who", "where", "does", "can you"},
    "empathetic":  {"sad", "depressed", "tired", "stressed", "anxious", "lonely", "miss", "hurt", "crying", "😢", "😔"},
}

_user_moods: dict = {}
_auto_mode: set = set()

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
            "\nReference these naturally when relevant."
        )

    if web_context:
        system += (
            f"\n\nReal-time web search results:\n{web_context}"
            "\nUse this to answer accurately."
        )

    # Split history: all but last message goes as history, last is the current turn
    if history:
        chat_history = history[:-1]
        current = history[-1]["content"]
    else:
        chat_history = []
        current = ""

    try:
        return gemini_call(system, chat_history, current, max_tokens=512)
    except Exception as e:
        err_str = str(e)
        print(f"[ai_chat error] {type(e).__name__}: {err_str}")
        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
            raise
        print(f"[ai_chat failed] {e}")
        return random.choice(FALLBACK_RESPONSES)
