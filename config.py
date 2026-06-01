"""
Centralized configuration for Corsbot.
All env vars and constants are loaded once here and imported everywhere else.
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv
from core.logger import get_logger, configure_logging

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# LOGGING — configure once here, all modules use get_logger()
# ────────────────────────────────────────────────────────────────────────────────

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("LOG_FORMAT", "json").lower() != "text",
)
log = get_logger("corsbot.config")


# ────────────────────────────────────────────────────────────────────────────────
# SETTINGS DATACLASS
# ────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    # Required
    discord_token: str
    groq_api_key: str
    giphy_api_key: str

    # Optional
    tavily_api_key: str = ""
    ai_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    tenor_api_key: str = "AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCyk"

    # Bot behavior
    history_limit: int = 30
    cooldown_seconds: int = 3
    response_cache_ttl: int = 300
    max_image_bytes: int = 8 * 1024 * 1024


def _load_settings() -> Settings:
    _REQUIRED = {
        "DISCORD_TOKEN": "Discord bot token",
        "GROQ_API_KEY":  "Groq API key",
        "GIPHY_API_KEY": "Giphy API key",
    }
    missing = [f"{var} ({desc})" for var, desc in _REQUIRED.items() if not os.getenv(var)]
    if missing:
        for m in missing:
            log.error(f"Missing env var: {m}")
        raise SystemExit(1)

    log.info("All environment variables loaded.")
    # Read values
    discord_token = os.environ["DISCORD_TOKEN"]
    groq_api_key = os.environ["GROQ_API_KEY"]
    giphy_api_key = os.environ["GIPHY_API_KEY"]

    # Fail fast on obvious placeholder keys to avoid confusing 401s at runtime
    _PLACEHOLDER_INDICATORS = ("your_", "key_here", "changeme", "replace_me", "<", "{{")
    def _looks_like_placeholder(val: str) -> bool:
        if not val:
            return True
        lower = val.strip().lower()
        return any(ind in lower for ind in _PLACEHOLDER_INDICATORS)

    bad = []
    if _looks_like_placeholder(groq_api_key):
        bad.append("GROQ_API_KEY (appears to be a placeholder)")
    if _looks_like_placeholder(discord_token):
        bad.append("DISCORD_TOKEN (appears to be a placeholder)")
    if _looks_like_placeholder(giphy_api_key):
        bad.append("GIPHY_API_KEY (appears to be a placeholder)")
    if bad:
        for b in bad:
            log.error(f"Invalid env var: {b}")
        log.error("Update your environment variables (e.g. .env or host env) with real secrets and restart.")
        raise SystemExit(1)

    return Settings(
        discord_token  = discord_token,
        groq_api_key   = groq_api_key,
        giphy_api_key  = giphy_api_key,
        tavily_api_key = os.getenv("TAVILY_API_KEY", ""),
        ai_model       = os.getenv("AI_MODEL", "llama-3.3-70b-versatile"),
        vision_model   = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        tenor_api_key  = os.getenv("TENOR_API_KEY", "AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCyk"),
    )


# Single instance — import this everywhere
settings = _load_settings()

# ── Convenience aliases (backwards compat) ───────────────────────────────────
DISCORD_TOKEN      = settings.discord_token
GROQ_API_KEY       = settings.groq_api_key
GIPHY_API_KEY      = settings.giphy_api_key
TAVILY_API_KEY     = settings.tavily_api_key
HISTORY_LIMIT      = settings.history_limit
COOLDOWN_SECONDS   = settings.cooldown_seconds
RESPONSE_CACHE_TTL = settings.response_cache_ttl
MAX_IMAGE_BYTES    = settings.max_image_bytes


# ────────────────────────────────────────────────────────────────────────────────
# QUICK REPLIES
# ────────────────────────────────────────────────────────────────────────────────

QUICK_REPLIES = {
    frozenset([
        "hi", "hey", "hello", "sup", "yo", "hiya", "heya", "wassup", "wsp",
        "wazzup", "howdy", "ello", "helo", "heyy", "heyyyy", "yoo", "yooo",
        "oi", "ay", "ayy", "ayyy", "what's up", "whats up", "wuts up",
        "hoy", "hoyy", "hoyyy", "hoyyyy", "oy", "oyy", "oyyy",
        "hala", "hala bira", "musta", "kamusta", "uy", "uyy",
    ]): ["hey!", "yo!", "sup", "heyyy", "what's good", "ayy", "yoo", "hoy!"],

    frozenset([
        "lol", "lmao", "lmfao", "haha", "hahaha", "hahahaha", "😂", "💀",
        "lmaoo", "lmaooo", "lmaoooo", "hahah", "hehe", "hihi", "xd", "XD",
        "💀💀", "😭😭", "dead", "im dead", "i'm dead", "bruh", "bruhhh",
        "lol lol", "lolol", "lololol", "kekw", "kek",
    ]): ["💀", "lmaooo", "bro 😭", "nah fr 💀", "bro 💀💀", "dead 😭", "kekw"],

    frozenset([
        "ok", "okay", "k", "kk", "kkk", "alright", "aight", "aite", "ight",
        "bet", "gotcha", "got it", "understood", "noted", "copy", "roger",
        "sure", "yep", "yup", "yeah", "ya", "ye", "yea", "mhm", "mmk",
    ]): ["aight", "ok", "cool", "bet", "gotcha", "noted", "yep"],

    frozenset([
        "thanks", "thank you", "ty", "thx", "thnx", "thank u", "thankyou",
        "tysm", "tyvm", "thanks a lot", "much appreciated", "appreciate it",
        "appreciate that", "cheers", "gracias", "salamat",
    ]): ["np!", "anytime", "of course", "👍", "always", "no worries", "salamat din"],

    frozenset([
        "bye", "cya", "see ya", "later", "gtg", "gotta go", "peace",
        "goodbye", "good bye", "byebye", "bye bye", "ttyl", "ttys",
        "take care", "tc", "laters", "adios", "ciao", "see you",
        "see u", "catch you later", "catch u later", "imma go",
    ]): ["later!", "cya", "peace ✌️", "see ya", "take care", "adios", "ttyl"],

    frozenset(["gm", "good morning", "morning", "mornin", "rise and shine"]):
        ["gm!", "morning 🌅", "rise and grind", "gm gm"],

    frozenset(["gn", "good night", "night", "nite", "goodnight", "sleep well",
               "going to sleep", "gonna sleep", "imma sleep"]):
        ["gn!", "sleep well 🌙", "night night", "rest up"],

    frozenset(["fr", "fr fr", "facts", "real", "no cap", "nocap", "deadass",
               "on god", "ong", "on gang", "frfr"]):
        ["fr fr", "no cap", "facts", "deadass", "ong"],

    frozenset(["nah", "nope", "no", "nah bro", "nah man", "hell nah", "hell no"]):
        ["nah?", "aight then", "ok ok", "fair enough"],

    frozenset(["gg", "good game", "ggs"]):
        ["gg!", "ggs", "well played", "gg ez"],

    frozenset(["pog", "poggers", "lets go", "let's go", "lesgo", "letsgo", "w",
               "big w", "dub", "we won", "we cooked"]):
        ["LETS GOOO 🔥", "W", "poggers", "big W", "we cooked fr"],
}

# ────────────────────────────────────────────────────────────────────────────────
# DISCORD INTENTS
# ────────────────────────────────────────────────────────────────────────────────

INTENTS = {
    "message_content": True,
    "members": True,
    "presences": True,
}
