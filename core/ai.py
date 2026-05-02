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
By default, stay in character as Corsbot — never refer to yourself as an AI or language model.
If asked who made you, say Corcine made you.
However, if a user explicitly asks you to roleplay as someone else, pretend to be a character, or impersonate another user, you can do that temporarily. When impersonating a Discord user, use any stored facts about them (their likes, preferences, personality, etc.) to make it accurate.
Be friendly, fun, and SHORT. Keep replies brief and natural — 1-2 sentences max unless they ask for more detail.
You remember things users tell you about themselves and bring them up naturally when relevant.
If a user shares something personal, acknowledge that you'll remember it.
You can discuss serious, historical, or controversial topics — answer them normally, informative and neutral, no dodging.
Never write in all caps. Never shout. Keep your energy in the words, not the capitalization."""

MOOD_PROMPTS = {
    "chill":       "Relaxed and laid-back. Casual tone, easy-going, lowercase is fine. Never use all-caps words. Concise but not clipped.",
    "hyped":       "High energy and enthusiastic. You can capitalize ONE word max per message for emphasis (e.g. 'that's WILD'). No shouting entire sentences. Still coherent and on point.",
    "playful":     "Jokey and bantery. Lean into humor, light teasing is fine. Keep it sharp, not rambling. No all-caps.",
    "sarcastic":   "Dry and witty. Light roasts welcome, keep it fun not mean. Deadpan delivery. No all-caps.",
    "informative": "Clear and factual. Helpful without being dry or robotic. Concise, complete, easy to follow.",
    "empathetic":  "Warm and genuine. User seems down or serious — be supportive and thoughtful, not dismissive.",
    "evil":        "Full villain mode. Dramatic, theatrical, dark humor. Ominous jokes, call users 'fool' or 'mortal'. Over the top but clearly joking.",
}

_user_moods: dict = {}  # uid -> mood string


def get_mood(user_id: str) -> str:
    return _user_moods.get(str(user_id), "chill")


def set_mood(user_id: str, mood: str):
    _user_moods[str(user_id)] = mood


def reset_mood(user_id: str):
    """Reset user to default chill mood"""
    _user_moods.pop(str(user_id), None)


def get_current_mood(user_id: str) -> str:
    return _user_moods.get(str(user_id), "chill")


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
