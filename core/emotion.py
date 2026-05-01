import random
import requests
import re
import os

from .ai import groq_call

GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

# ---------------- FORCE TRIGGERS ---------------- #
# Only strong, unambiguous signals — no single common words

# Exact-match triggers (must be the whole word/phrase)
FORCE_EMOTION_TRIGGERS = {
    "laugh":     {"lol", "lmao", "lmfao", "hahaha", "💀", "😂", "😭", "im dead", "i'm dead"},
    "celebrate": {"congrats", "congratulations", "happy birthday", "hbd", "🎉", "🥳", "🎂"},
    "sad":       {"rip", "f in chat", "😢", "😔", "💔"},
    "shock":     {"wtf", "omg", "no way bro", "bro what", "😱", "🤯", "wait what"},
    "angry":     {"😤", "😡", "🤬"},
    "happy":     {"yay", "lets gooo", "let's gooo", "poggers", "🥰", "😍"},
}

GIF_MAP = {
    "laugh":     ["laughing hysterically", "dying laughing", "that's hilarious", "can't stop laughing"],
    "happy":     ["happy dance", "excited reaction", "yay celebration"],
    "sad":       ["sad crying", "rip moment", "f in chat"],
    "angry":     ["angry reaction", "rage quit", "furious"],
    "shock":     ["shocked reaction", "mind blown", "jaw drop"],
    "celebrate": ["party celebration", "confetti", "birthday cake"],
}

GIF_CHANCE = {
    "laugh":     0.25,
    "celebrate": 0.20,
    "shock":     0.12,
    "happy":     0.08,
    "sad":       0.07,
    "angry":     0.06,
}

def detect_forced_emotion(text: str):
    """Only match strong unambiguous triggers using word boundaries for short tokens."""
    lower = text.lower().strip()
    for emotion, triggers in FORCE_EMOTION_TRIGGERS.items():
        for trigger in triggers:
            if len(trigger) <= 4:
                # Short tokens: require word boundary so "lol" doesn't match "lolwut"
                if re.search(rf"\b{re.escape(trigger)}\b", lower):
                    return emotion
            else:
                # Longer phrases: substring match is fine
                if trigger in lower:
                    return emotion
    return None

def build_gif_query(emotion: str, context: str) -> str:
    base_queries = GIF_MAP.get(emotion, [emotion])
    return random.choice(base_queries)

def pick_gif_for_message(content: str, reply: str) -> tuple:
    """Only send a GIF when there's a strong forced emotion signal. No AI detection."""
    forced = detect_forced_emotion(content)
    if forced and random.random() < GIF_CHANCE.get(forced, 0.20):
        from .gif import search_gif
        url = search_gif(build_gif_query(forced, content))
        if url:
            return url, forced
    return None, None
