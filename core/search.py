import os
import re
import time

import aiohttp

from config import settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"
DDG_INSTANT_URL = "https://api.duckduckgo.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 300

REALTIME_TRIGGERS = {
    "news",
    "latest",
    "today's",
    "right now",
    "live score",
    "scores",
    "standings",
    "match result",
    "breaking",
    "breaking news",
    "what's happening",
    "whats happening",
    "who won",
    "game score",
    "match score",
    "final score",
    "weather",
    "temperature",
    "forecast",
    "stock price",
    "crypto price",
    "bitcoin price",
    "ethereum price",
    "exchange rate",
    "patch notes",
    "new update",
    "new patch",
    "new season",
    "new episode",
    "just dropped",
    "just announced",
    "just released",
    "just launched",
    "release date",
    "when does",
    "when is",
    "what happened to",
    "what's the latest",
    "whats the latest",
    "is it out",
    "is it available",
    "is it live",
    "who is",
    "who's",
    "whos",
}

SONG_QUERY_TRIGGERS = {
    "what song",
    "what's the song",
    "whats the song",
    "what's that song",
    "whats that song",
    "guess the song",
    "identify the song",
    "name the song",
    "name that tune",
    "tell me the song",
    "based on the lyrics",
    "from the lyrics",
    "these lyrics",
    "song is this",
    "that song",
    "this song",
}

SONG_FOLLOWUP_TRIGGERS = {
    "take another guess",
    "another guess",
    "guess again",
    "one more guess",
    "one more line",
    "another line",
    "more lyrics",
    "repeat the lyrics",
    "song by",
    "artist is",
    "who sang this",
    "who sings this",
    "who sang it",
    "who sings it",
    "it is song",
    "it's a song",
    "its a song",
    "yes it is song",
    "be serious",
    "bot be serious",
    "not it",
    "not that one",
    "not this one",
}

_ARTIST_PATTERNS = (
    re.compile(r"\bsong by\s+([a-z0-9 '&.,-]{2,60})", re.I),
    re.compile(r"\bartist is\s+([a-z0-9 '&.,-]{2,60})", re.I),
    re.compile(r"\bby\s+([a-z0-9 '&.,-]{2,60})\.?$", re.I),
)


def _clean_history_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"^\[[^\]]+\]:\s*", "", text).strip()


def _is_artist_hint(text: str) -> bool:
    cleaned = _clean_history_text(text).strip().strip("\"'`")
    if not cleaned:
        return False
    if any(pattern.search(cleaned) for pattern in _ARTIST_PATTERNS):
        return True
    lower = cleaned.lower()
    return lower.startswith("artist:") or lower.startswith("artist -")


def _recent_user_lines(history: list[dict] | None, limit: int = 10) -> list[str]:
    lines = []
    for entry in history or []:
        if entry.get("role") != "user":
            continue
        text = _clean_history_text(entry.get("content", ""))
        if text:
            lines.append(text)
    return lines[-limit:]


def _is_artist_hint(text: str) -> bool:
    cleaned = _clean_history_text(text).strip().strip("\"'`")
    if not cleaned:
        return False
    if any(pattern.search(cleaned) for pattern in _ARTIST_PATTERNS):
        return True
    lower = cleaned.lower()
    return lower.startswith("artist:") or lower.startswith("artist -")


def _looks_like_lyric_fragment(text: str) -> bool:
    cleaned = _clean_history_text(text).strip().strip("\"'`")
    lower = cleaned.lower()
    word_count = len(lower.split())
    if word_count < 2 or word_count > 10:
        return False
    if lower.endswith("?"):
        return False
    if any(
        trigger in lower for trigger in SONG_QUERY_TRIGGERS | SONG_FOLLOWUP_TRIGGERS
    ):
        return False
    if re.search(r"https?://|/play\b|<@!?(\d+)>", lower):
        return False
    alpha_chars = sum(ch.isalpha() for ch in lower)
    return alpha_chars >= 6


def merge_song_clue_lines(*clue_lists: list[str], max_lines: int = 8) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for clues in clue_lists:
        for line in clues or []:
            normalized = line.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(line.strip())
    return merged[-max_lines:]


def is_song_followup_message(message: str) -> bool:
    lower = (message or "").lower().strip()
    if any(trigger in lower for trigger in SONG_FOLLOWUP_TRIGGERS):
        return True
    if lower in {
        "yes",
        "yeah",
        "yep",
        "nah",
        "nope",
        "no",
        "not it",
        "not that one",
        "not this one",
    }:
        return True
    return _looks_like_lyric_fragment(message)


def extract_song_artist_hint(message: str, history: list[dict] | None = None) -> str:
    for line in [message] + _recent_user_lines(history, limit=12):
        if _is_artist_hint(line):
            for pattern in _ARTIST_PATTERNS:
                match = pattern.search(line)
                if match:
                    return match.group(1).strip()
            cleaned = _clean_history_text(line).strip().strip("\"'`")
            if cleaned.lower().startswith("artist:"):
                return cleaned.split(":", 1)[1].strip()
            if cleaned.lower().startswith("artist -"):
                return cleaned.split("-", 1)[1].strip()
    return ""


def build_song_search_query_from_clues(
    clue_lines: list[str], artist_hint: str = ""
) -> str:
    parts = ["lyrics"]
    parts.extend(f'"{line}"' for line in clue_lines if line)
    if artist_hint:
        parts.append(f'"{artist_hint}"')
    if len(parts) == 1:
        return ""
    return " ".join(parts)[:240]


def needs_web_search(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in REALTIME_TRIGGERS)


def is_song_identification_turn(
    message: str, history: list[dict] | None = None
) -> bool:
    lower = message.lower().strip()
    if any(trigger in lower for trigger in SONG_QUERY_TRIGGERS):
        return True

    recent_lines = _recent_user_lines(history)
    recent_has_song_task = any(
        any(
            trigger in line.lower()
            for trigger in SONG_QUERY_TRIGGERS | SONG_FOLLOWUP_TRIGGERS
        )
        for line in recent_lines[-6:]
    )
    if not recent_has_song_task:
        return False

    if any(trigger in lower for trigger in SONG_FOLLOWUP_TRIGGERS):
        return True
    if lower in {
        "yes",
        "yeah",
        "yep",
        "nah",
        "nope",
        "no",
        "not it",
        "not that one",
        "not this one",
    }:
        return True
    return _looks_like_lyric_fragment(message)


def collect_song_clues(
    message: str, history: list[dict] | None = None, max_lines: int = 6
) -> list[str]:
    clues = []
    seen = set()
    candidates = [message] + _recent_user_lines(history, limit=16)
    for line in candidates:
        cleaned = _clean_history_text(line).strip().strip("\"'`")
        lower = cleaned.lower()
        if not cleaned:
            continue
        if any(
            trigger in lower for trigger in SONG_QUERY_TRIGGERS | SONG_FOLLOWUP_TRIGGERS
        ):
            continue
        if lower in {
            "yes",
            "yeah",
            "yep",
            "nah",
            "nope",
            "no",
            "not it",
            "not that one",
            "not this one",
        }:
            continue
        if _is_artist_hint(cleaned):
            continue
        if not _looks_like_lyric_fragment(cleaned):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        clues.append(cleaned)
    return clues[-max_lines:]


def build_song_search_query(message: str, history: list[dict] | None = None) -> str:
    artist_hint = extract_song_artist_hint(message, history)
    clue_lines = collect_song_clues(message, history, max_lines=4)
    query = build_song_search_query_from_clues(clue_lines, artist_hint)
    return query or build_search_query(message)


async def ddg_instant(session: aiohttp.ClientSession, query: str) -> str | None:
    try:
        async with session.get(
            DDG_INSTANT_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json(content_type=None)
        if data.get("AbstractText"):
            return data["AbstractText"][:600]
        topics = data.get("RelatedTopics", [])
        snippets = [
            t["Text"] for t in topics[:3] if isinstance(t, dict) and t.get("Text")
        ]
        if snippets:
            return " | ".join(snippets)[:600]
    except Exception as e:
        print(f"[search] DDG instant failed: {e}")
    return None


async def tavily_search(
    session: aiohttp.ClientSession, query: str, max_results: int = 3
) -> str | None:
    if not settings.tavily_api_key:
        return None
    try:
        async with session.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "search_depth": "basic",
            },
            timeout=aiohttp.ClientTimeout(total=12),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        results = data.get("results", [])
        if not results:
            return None
        clean = []
        for item in results[:max_results]:
            title = item.get("title", "").strip()
            url = item.get("url") or item.get("link") or ""
            snippet = re.sub(
                r"\s+", " ", item.get("content", "") or item.get("raw_content", "")
            ).strip()
            if title and snippet:
                clean.append(f"{title} — {snippet} [source: {url}]")
            elif snippet:
                clean.append(f"{snippet} [source: {url}]")
        return "\n".join(clean)[:1200] if clean else None
    except Exception as e:
        print(f"[search] Tavily failed: {e}")
    return None


async def ddg_search(
    session: aiohttp.ClientSession, query: str, max_results: int = 3
) -> str | None:
    try:
        async with session.post(
            DDG_SEARCH_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            html = await resp.text()
        results = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        clean = []
        for idx, (link, raw_title) in enumerate(results[:max_results]):
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = (
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", snippets[idx])).strip()
                if idx < len(snippets)
                else ""
            )
            if title or snippet:
                clean.append(
                    f"{title} — {snippet} [source: {link}]"
                    if snippet
                    else f"{title} [source: {link}]"
                )
        return "\n".join(clean)[:1200] if clean else None
    except Exception as e:
        print(f"[search] DDG HTML failed: {e}")
    return None


async def web_search(query: str) -> str | None:
    key = query.lower().strip()
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return result

    result = None
    async with aiohttp.ClientSession() as session:
        instant = await ddg_instant(session, query)
        if instant:
            result = f"DuckDuckGo: {instant}"
        else:
            tavily = await tavily_search(session, query)
            if tavily:
                result = f"Tavily: {tavily}"
            else:
                result = await ddg_search(session, query)

    if result:
        _cache[key] = (result, time.time())
    return result


def build_search_query(message: str) -> str:
    cleaned = re.sub(r"\b(corsbot|hey|please|bro|can you)\b", "", message.lower())
    return re.sub(r"\s+", " ", cleaned).strip()[:200]
