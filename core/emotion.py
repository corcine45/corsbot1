import random
import requests
import os

from .ai import groq_call

GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

# ---------------- FORCE TRIGGERS ---------------- #

FORCE_EMOTION_TRIGGERS = {
    "laugh":     {"lol", "lmao", "lmfao", "haha", "hahaha", "💀", "😂", "😭", "bruh", "dead", "im dead"},
    "celebrate": {"congrats", "congratulations", "happy birthday", "hbd", "lets go", "let's go", "🎉", "🥳", "🎂", "w", "big w"},
    "sad":       {"rip", "f in chat", "😢", "😔", "💔", "oof", "that hurts"},
    "shock":     {"no way", "what", "wtf", "omg", "bro what", "😱", "🤯", "wait what"},
    "angry":     {"😤", "😡", "🤬", "trash", "garbage", "scam"},
    "happy":     {"yay", "wooo", "lets gooo", "poggers", "pog", "🥰", "😍"},
}

GIF_MAP = {
    "laugh":     ["laughing hysterically", "dying laughing", "that's hilarious", "can't stop laughing"],
    "happy":     ["happy dance", "excited reaction", "yay celebration"],
    "sad":       ["sad crying", "rip moment", "f in chat"],
    "angry":     ["angry reaction", "rage quit", "furious"],
    "shock":     ["shocked reaction", "mind blown", "no way reaction", "jaw drop"],
    "celebrate": ["party celebration", "confetti", "birthday cake", "lets go reaction"],
}

GIF_CHANCE = {
    "laugh":     0.45,
    "celebrate": 0.40,
    "shock":     0.25,
    "happy":     0.15,
    "sad":       0.12,
    "angry":     0.10,
}

def detect_forced_emotion(text: str):
    lower = text.lower()
    for emotion, triggers in FORCE_EMOTION_TRIGGERS.items():
        for trigger in triggers:
            if trigger in lower:
                return emotion
    return None

def ai_detect_emotions(text: str) -> list:
    try:
        raw = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Detect the emotions in this text. "
                    "Reply with 1 or 2 words from: happy, sad, angry, laugh, shock, celebrate, none. "
                    "If multiple, separate with comma. Example: laugh,shock"
                )},
                {"role": "user", "content": text}
            ],
            max_tokens=15, retries=2, timeout=8,
        )
        emotions = [e.strip() for e in raw.lower().split(",") if e.strip() in GIF_MAP]
        return emotions if emotions else ["none"]
    except:
        return ["none"]

def build_gif_query(emotion: str, context: str) -> str:
    base_queries = GIF_MAP.get(emotion, [emotion])
    keywords = [w for w in context.lower().split() if len(w) > 4]
    if keywords and random.random() < 0.4:
        return f"{random.choice(base_queries)} {random.choice(keywords[:5])}"
    return random.choice(base_queries)

def pick_gif_for_message(content: str, reply: str) -> tuple:
    combined = (content + " " + reply)[:300]
    forced = detect_forced_emotion(content)
    if forced:
        from .gif import search_gif
        url = search_gif(build_gif_query(forced, content))
        return url, forced

    emotions = ai_detect_emotions(combined)
    for emotion in emotions:
        if emotion in GIF_MAP and random.random() < GIF_CHANCE.get(emotion, 0.10):
            from .gif import search_gif
            url = search_gif(build_gif_query(emotion, combined))
            if url:
                return url, emotion
    return None, None
