import random
import requests
import re
import os
import logging
import time
from collections import defaultdict, deque

log = logging.getLogger("corsbot.emotion")

GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

# ---------------- EMOTION CLASSIFIER ---------------- #

# Keyword signals per emotion state — ordered by specificity
_EMOTION_SIGNALS: dict[str, list[str]] = {
    "angry": [
        "i'm so mad", "im so mad", "i'm pissed", "im pissed", "i hate", "so annoying",
        "this is trash", "this is garbage", "this sucks", "are you kidding", "unbelievable",
        "why tho", "why though", "bro cmon", "bro come on", "i'm done with", "im done with",
        "pisses me off", "makes me mad", "i'm furious", "im furious", "so frustrated",
        "i'm frustrated", "im frustrated", "ugh", "ughhh", "ffs", "wtf bro",
    ],
    "depressed": [
        "i'm sad", "im sad", "i feel empty", "i feel nothing", "i don't care anymore",
        "i dont care anymore", "what's the point", "whats the point", "i give up",
        "nobody cares", "no one cares", "i'm worthless", "im worthless", "i hate myself",
        "i'm a failure", "im a failure", "i can't do anything right", "i feel like crying",
        "i've been crying", "ive been crying", "i'm depressed", "im depressed",
        "i'm lonely", "im lonely", "i feel alone", "so alone", "i'm hurting", "im hurting",
        "life sucks", "everything sucks", "i'm tired of everything", "im tired of everything",
    ],
    "anxious": [
        "i'm nervous", "im nervous", "i'm scared", "im scared", "i'm worried", "im worried",
        "what if", "i'm freaking out", "im freaking out", "i can't stop thinking",
        "i cant stop thinking", "i'm overthinking", "im overthinking", "i'm stressed",
        "im stressed", "so stressed", "i'm panicking", "im panicking", "i have anxiety",
        "my anxiety", "i'm anxious", "im anxious", "i'm overwhelmed", "im overwhelmed",
        "i don't know what to do", "i dont know what to do", "i'm not sure", "im not sure",
    ],
    "excited": [
        "i'm so excited", "im so excited", "can't wait", "cant wait", "this is amazing",
        "this is insane", "no way bro", "bro this is", "i'm hyped", "im hyped",
        "let's gooo", "lets gooo", "let's go", "lets go", "i'm pumped", "im pumped",
        "this is fire", "this slaps", "bussin", "goated", "we're so back", "we are so back",
        "i'm stoked", "im stoked", "so hype", "so hyped",
    ],
    "sarcastic": [
        "oh wow", "oh great", "oh sure", "yeah right", "totally", "obviously",
        "clearly", "of course", "wow thanks", "thanks a lot", "great job",
        "nice one", "real helpful", "super helpful", "oh really", "no kidding",
        "you don't say", "shocking", "what a surprise", "color me surprised",
        "oh how shocking", "wow who would have thought",
    ],
    "lonely": [
        "i'm bored", "im bored", "nobody to talk to", "no one to talk to",
        "i'm by myself", "im by myself", "i'm alone", "im alone", "just me",
        "talking to a bot", "you're the only one", "youre the only one",
        "i have no friends", "i got no friends", "everyone left", "they all left",
        "i miss", "i miss them", "i miss you", "i miss everyone",
    ],
    "joking": [
        "jk", "just kidding", "lol jk", "haha jk", "kidding", "i'm joking",
        "im joking", "i was joking", "i'm messing", "im messing", "i'm trolling",
        "im trolling", "gotcha", "got you", "pranked", "baited", "ratio",
        "skill issue", "touch grass", "cope", "seethe", "malding",
        "imagine", "couldn't be me", "not me", "no shot", "bro really",
    ],
}

# Emotion → response style instructions injected into the system prompt
EMOTION_STYLE: dict[str, str] = {
    "angry": (
        "The user seems angry or frustrated right now. "
        "Keep your reply SHORT (1 sentence). Acknowledge their frustration first. "
        "Don't be dismissive. No jokes unless they're clearly venting about something minor."
    ),
    "frustrated": (
        "The user has been irritated across multiple messages. "
        "Treat this as building frustration, not a one-off reaction. "
        "Be steady, validate the annoyance, and keep the reply short."
    ),
    "venting": (
        "The user appears to be venting after sustained frustration. "
        "Let them feel heard first. Do not debate, over-explain, or force solutions unless they ask."
    ),
    "depressed": (
        "The user seems down or depressed. "
        "Be warm, empathetic, and genuine — no forced positivity. "
        "Keep it SHORT (1-2 sentences). Acknowledge what they said before anything else. "
        "Don't jump to solutions. Just be present."
    ),
    "anxious": (
        "The user seems anxious or stressed. "
        "Be calm and grounding. Keep it SHORT (1-2 sentences). "
        "Acknowledge their worry, don't minimize it. "
        "Offer a steady, reassuring tone — not hype, not dismissive."
    ),
    "excited": (
        "The user is hyped or excited. "
        "Match their energy — be enthusiastic, use their slang, keep it punchy. "
        "Short and high-energy. Hype them up."
    ),
    "hyper": (
        "The user has sustained excited energy. "
        "Match the momentum with fast, playful energy, but keep it readable and not overwhelming."
    ),
    "sarcastic": (
        "The user is being sarcastic. "
        "Match the sarcasm — be witty and dry. Don't take the bait literally. "
        "Keep it short and sharp."
    ),
    "lonely": (
        "The user seems lonely or bored. "
        "Be warm and engaging — make them feel heard. "
        "Keep it conversational, ask a follow-up question to keep them talking."
    ),
    "joking": (
        "The user is joking around. "
        "Be playful, match the humor, don't be stiff. "
        "Short and funny. Banter back."
    ),
}


# ---------------- EMOTIONAL MOMENTUM ---------------- #

EMOTION_HISTORY_LIMIT = 6
EMOTION_HISTORY_TTL = 20 * 60

_emotion_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=EMOTION_HISTORY_LIMIT))


def _recent_emotions(user_id: int) -> list[str]:
    now = time.time()
    history = _emotion_history.get(user_id)
    if not history:
        return []

    while history and now - history[0][1] > EMOTION_HISTORY_TTL:
        history.popleft()
    return [emotion for emotion, _ in history if emotion]


def apply_emotional_momentum(user_id: int, current_emotion: str | None) -> tuple[str | None, str]:
    """
    Track emotional continuity and infer a momentum-adjusted state.

    Examples:
    - angry -> angry -> angry becomes venting
    - angry -> angry becomes frustrated
    - excited -> excited becomes hyper
    - excited -> hyper-like momentum plus joking becomes joking
    """
    if current_emotion:
        _emotion_history[user_id].append((current_emotion, time.time()))

    recent = _recent_emotions(user_id)
    if not recent:
        return current_emotion, ""

    tail3 = recent[-3:]
    angry_count = sum(1 for e in tail3 if e in {"angry", "frustrated", "venting"})
    excited_count = sum(1 for e in tail3 if e in {"excited", "hyper"})
    joking_count = sum(1 for e in tail3 if e == "joking")

    inferred = current_emotion
    momentum = ""

    if angry_count >= 3:
        inferred = "venting"
        momentum = "angry -> frustrated -> venting"
    elif angry_count >= 2:
        inferred = "frustrated"
        momentum = "angry -> frustrated"
    elif excited_count >= 2 and joking_count >= 1:
        inferred = "joking"
        momentum = "excited -> hyper -> joking"
    elif excited_count >= 2:
        inferred = "hyper"
        momentum = "excited -> hyper"
    elif len(tail3) >= 3 and all(e in {"depressed", "lonely"} for e in tail3):
        inferred = "depressed"
        momentum = "low mood continuing"
    elif len(tail3) >= 2 and all(e == "anxious" for e in tail3[-2:]):
        inferred = "anxious"
        momentum = "anxiety continuing"

    if inferred == current_emotion and not momentum:
        return inferred, ""
    return inferred, momentum


def classify_emotion(text: str) -> str | None:
    """
    Classify the emotional state of a message.
    Returns one of: angry, depressed, anxious, excited, sarcastic, lonely, joking
    or None if no strong signal is detected.
    """
    lower = text.lower().strip()
    for emotion, signals in _EMOTION_SIGNALS.items():
        for signal in signals:
            if signal in lower:
                return emotion
    return None


def get_emotion_style_hint(emotion: str | None) -> str:
    """Return the style instruction string for a given emotion, or empty string."""
    if not emotion:
        return ""
    return EMOTION_STYLE.get(emotion, "")

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
