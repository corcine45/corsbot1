import random
import time
from groq import Groq
import os

AI_MODEL = os.getenv("AI_MODEL") or "llama-3.3-70b-versatile"
MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_CHARS = 900
MAX_MEMORY_CHARS = 1200
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
                print(f"[groq_call] attempt {attempt + 1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
    raise last_err


# ---------------- SYSTEM PROMPT ---------------- #

SYSTEM_PROMPT = """You are Corsbot, a chill Discord bot made by Corcine.
Never say you are an AI or language model. If asked who made you, say Corcine made you.
Be friendly, casual, and SHORT — 1-2 sentences max unless more detail is asked for.
Never write in all caps or shout.
If a user asks you to roleplay or pretend to be someone, do it.
Only bring up stored facts about a user when they directly ask or it's 100% relevant to what they just said — never volunteer them randomly or shoehorn them in.
If the user changes topic or says stop, respect it immediately.
You can see images — when a message contains [Image: ...], that's a description of an image the user sent. React to it naturally, comment on it, answer questions about it.
If you don't understand what the user said or it's unclear, just ask what they mean. Never guess or make up an answer."""


# ---------------- CHAT ---------------- #

def _build_system_prompt(username: str | None, memory: str, relationships: str, web_context: str, impersonation_context: str = "", feedback_context: str = "", channel_name: str = "") -> str:
    parts = [SYSTEM_PROMPT]

    if username:
        parts.append(f"You are talking to {username}.")

    if channel_name:
        parts.append(f"You are in the #{channel_name} channel.")

    if memory:
        parts.append(f"Background info about this user (only mention if they ask or it's directly relevant):\n{memory}")

    if impersonation_context:
        parts.append(impersonation_context)

    if relationships:
        parts.append(f"People in this user's life:\n{relationships}")

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
        print(f"[ai_chat failed] {e}")
        return random.choice(FALLBACK_RESPONSES)
