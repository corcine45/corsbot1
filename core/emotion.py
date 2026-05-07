import random
import requests
import re
import os

GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

# ---------------- FORCE TRIGGERS ---------------- #
# Only strong, unambiguous signals — no single common words

# Exact-match triggers (must be the whole word/phrase)
FORCE_EMOTION_TRIGGERS = {
    "laugh": {
        "lol", "lmao", "lmfao", "hahaha", "hahahaha", "💀", "😂", "😭",
        "im dead", "i'm dead", "dead", "lmaoo", "lmaooo", "lmaoooo",
        "kekw", "kek", "lolol", "lololol", "crying laughing", "i cant",
        "i can't", "bro 💀", "💀💀", "😭😭", "bruh moment", "not me",
        "i'm crying", "im crying", "this is hilarious", "i'm done",
        "im done", "i'm weak", "im weak", "sent me", "i'm deceased",
    },
    "celebrate": {
        "congrats", "congratulations", "happy birthday", "hbd", "🎉", "🥳", "🎂",
        "happy bday", "happy b-day", "feliz cumpleaños", "maligayang bati",
        "you did it", "we did it", "lets gooo", "let's gooo", "we won",
        "big w", "dub", "we cooked", "we ate", "we're so back",
        "we are so back", "he's back", "she's back", "they're back",
        "🏆", "🥇", "🎊", "🎈", "🎁",
    },
    "sad": {
        "rip", "f in chat", "😢", "😔", "💔", "f", "moment of silence",
        "that's sad", "thats sad", "that hurts", "ouch", "damn that's rough",
        "damn thats rough", "L", "big L", "took an L", "😞", "😟",
        "that's rough", "thats rough", "pour one out", "🪦", "gone too soon",
        "rest in peace", "rest easy", "we miss you", "miss you",
    },
    "shock": {
        "wtf", "omg", "no way bro", "bro what", "😱", "🤯", "wait what",
        "no way", "bro no way", "what the", "what the hell", "what the heck",
        "are you serious", "you serious", "seriously", "no shot", "no cap",
        "bro stop", "stop it", "shut up", "shut up bro", "no way fr",
        "wait fr", "wait actually", "hold on", "hold up", "wait wait wait",
        "WHAT", "NAH", "BRO", "BRUH", "😳", "👀", "👁️👄👁️",
    },
    "angry": {
        "😤", "😡", "🤬", "i'm mad", "im mad", "i'm pissed", "im pissed",
        "that's annoying", "thats annoying", "so annoying", "bro cmon",
        "bro come on", "are you kidding", "you kidding me", "unbelievable",
        "this is trash", "this is garbage", "this sucks", "i hate this",
        "i hate it", "why tho", "why though",
    },
    "happy": {
        "yay", "lets gooo", "let's gooo", "poggers", "🥰", "😍",
        "so happy", "i'm happy", "im happy", "this is great", "love this",
        "love it", "this is amazing", "amazing", "incredible", "goated",
        "this slaps", "it slaps", "bussin", "bussin fr", "fire", "🔥",
        "W", "big W", "pog", "gg", "good game", "we won", "🚀",
        "i'm so excited", "im so excited", "can't wait", "cant wait",
    },
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
