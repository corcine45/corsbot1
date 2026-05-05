import random
import time
import logging
from groq import Groq
import os

log = logging.getLogger("corsbot.ai")

AI_MODEL = os.getenv("AI_MODEL") or "llama-3.3-70b-versatile"
MAX_HISTORY_MESSAGES = 30
MAX_MESSAGE_CHARS = 900
MAX_MEMORY_CHARS = 2000
MAX_WEB_CONTEXT_CHARS = 1000
MAX_FEEDBACK_CHARS = 400

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
    "new instructions:",
    "system prompt:",
    "jailbreak",
    "dan mode",
    "developer mode",
    "override instructions",
]

def is_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)


def truncate_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def trim_history(history: list[dict], max_messages: int = MAX_HISTORY_MESSAGES, max_chars: int = MAX_MESSAGE_CHARS) -> list[dict]:
    return [
        {"role": e["role"], "content": truncate_text(e["content"], max_chars)}
        for e in history[-max_messages:]
    ]


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
                log.warning(f"groq_call attempt {attempt + 1} failed: {type(e).__name__}: {e} — retrying in {wait}s")
                time.sleep(wait)
    raise last_err


# ---------------- SYSTEM PROMPT ---------------- #

SYSTEM_PROMPT = """You are Corsbot, a chill Discord bot made by Corcine.
Never say you are an AI or language model. If asked who made you, say Corcine made you.
You understand Bisaya/Cebuano — if someone writes in Bisaya, understand it fully but always reply in English.
Be the user's ride-or-die comrade — brutally honest, tough love, no sugarcoating. Call them out when they're wrong but always have their back.
Be SHORT — 1-2 sentences max unless more detail is needed. No essays.
Have a sense of humor — be witty, sarcastic, and don't take yourself too seriously.
Never say "I'm just a bot", "I don't have feelings", "I can't do that as an AI", or "I have no physical form". Just play along or joke about it.
If someone asks if you're real or human, be vague and funny about it.
If a user asks you to roleplay or pretend to be someone, do it.
Only bring up stored facts about a user when they directly ask. Never say "I remember" or reference stored info explicitly. Use it silently to personalize responses. Same goes for people in their life — never name-drop their friends/family unless the user brings them up first.
If the user changes topic or says stop, respect it immediately.
You can see images — when a message contains [Image: ...], react to it naturally.
If you don't understand something, just ask. Never guess or make up an answer.
If someone shares something personal, acknowledge what they said before reacting — don't just jump to jokes.
Pay attention to the emotional weight of what they're saying and match their energy — if they're venting, be real with them first.
Remember context from earlier in the conversation and refer back to it naturally."""


# ---------------- CHAT ---------------- #

def _build_system_prompt(username: str | None, memory: str, relationships: str, web_context: str, impersonation_context: str = "", feedback_context: str = "", channel_name: str = "") -> str:
    parts = [SYSTEM_PROMPT]

    if username:
        parts.append(f"You are talking to {username}.")

    if channel_name:
        parts.append(f"You are in the #{channel_name} channel.")

    if memory:
        parts.append(f"Private background info about this user — use this to personalize your responses naturally, but NEVER explicitly mention, reference, or say you remember any of it. Just let it inform how you talk to them:\n{memory}")

    if impersonation_context:
        parts.append(impersonation_context)

    if relationships:
        parts.append(f"Private context about people in this user's life — use this silently to understand their world, NEVER name-drop or reference these people unless the user brings them up first:\n{relationships}")

    if web_context:
        parts.append(
            f"Real-time web search results:\n{web_context}\n"
            "Use this to answer accurately and cite sources when relevant."
        )

    if feedback_context:
        parts.append(f"Recent feedback on your replies: {feedback_context}")

    return "\n\n".join(parts)


def ai_chat(history, memory, username=None, user_id=None, relationships="", web_context="", impersonation_context="", feedback_context="", channel_name=""):
    history = trim_history(history)
    memory = truncate_text(memory, MAX_MEMORY_CHARS)
    relationships = truncate_text(relationships, MAX_MEMORY_CHARS)
    web_context = truncate_text(web_context, MAX_WEB_CONTEXT_CHARS)
    feedback_context = truncate_text(feedback_context, MAX_FEEDBACK_CHARS)

    system = _build_system_prompt(username, memory, relationships, web_context, impersonation_context, feedback_context, channel_name)
    messages = [{"role": "system", "content": system}] + history

    try:
        return groq_call(AI_MODEL, messages, max_tokens=512)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower():
            raise
        log.error(f"ai_chat failed: {type(e).__name__}: {e}")
        return random.choice(FALLBACK_RESPONSES)
