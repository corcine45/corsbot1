import os
import requests
import re
import time

# Tavily Search API (optional)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# DuckDuckGo instant answer + HTML search — no API key required
DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"
DDG_INSTANT_URL = "https://api.duckduckgo.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 300  # 5 minutes

# Keywords that signal the user wants real-time info
REALTIME_TRIGGERS = {
    "news", "latest", "today", "right now", "currently", "score", "scores",
    "standings", "match", "game", "result", "results", "update", "updates",
    "happening", "current", "recent", "just", "now", "live", "breaking",
    "weather", "price", "stock", "crypto", "bitcoin", "release", "dropped",
    "announced", "trailer", "patch", "patch notes", "season", "episode",
    "who won", "did they", "what happened", "when is", "when does",
}

def needs_web_search(text: str) -> bool:
    """Returns True if the message likely needs real-time information."""
    lower = text.lower()
    if any(trigger in lower for trigger in REALTIME_TRIGGERS):
        return True

    question_patterns = [
        "who is",
        "is it",
        "are they",
        "did they",
        "what's the",
        "how much is",
    ]
    return any(pattern in lower for pattern in question_patterns)


def ddg_instant(query: str) -> str | None:
    """Try DuckDuckGo instant answers first (fast, structured)."""
    try:
        res = requests.get(
            DDG_INSTANT_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers=HEADERS,
            timeout=5,
        )
        data = res.json()
        # AbstractText is the best source — Wikipedia-style summary
        if data.get("AbstractText"):
            return data["AbstractText"][:600]
        # RelatedTopics as fallback
        topics = data.get("RelatedTopics", [])
        snippets = []
        for t in topics[:3]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"])
        if snippets:
            return " | ".join(snippets)[:600]
        return None
    except Exception as e:
        print(f"[search] DDG instant failed: {e}")
        return None


def tavily_search(query: str, max_results: int = 3, search_depth: str = "basic") -> str | None:
    """Use Tavily search if a key is configured."""
    if not TAVILY_API_KEY:
        return None

    try:
        res = requests.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "search_depth": search_depth,
                "topic": "general",
            },
            timeout=12,
        )
        data = res.json()
        if res.status_code != 200:
            print(f"[search] Tavily failed: {res.status_code} {data}")
            return None

        results = data.get("results", [])
        if not results:
            return None

        clean = []
        for item in results[:max_results]:
            title = item.get("title", "").strip()
            url = item.get("url") or item.get("link") or item.get("source")
            snippet = item.get("content", "") or item.get("raw_content", "")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if title and snippet:
                clean.append(f"{title} — {snippet} [source: {url}]")
            elif title:
                clean.append(f"{title} [source: {url}]")
            elif snippet:
                source = f" [source: {url}]" if url else ""
                clean.append(f"{snippet}{source}")
        return "\n".join(clean)[:1200] if clean else None
    except Exception as e:
        print(f"[search] Tavily search failed: {e}")
        return None


def ddg_search(query: str, max_results: int = 3) -> str | None:
    """Scrape DuckDuckGo HTML search for snippets with source citations."""
    try:
        res = requests.post(
            DDG_SEARCH_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=8,
        )
        html = res.text

        # Extract titles, links, and snippets from result cards.
        results = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        clean = []
        for idx, (link, raw_title) in enumerate(results[:max_results]):
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = ""
            if idx < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[idx]).strip()
                snippet = re.sub(r"\s+", " ", snippet)
            if title or snippet:
                text = title
                if snippet:
                    text += f" — {snippet}"
                text += f" [source: {link}]"
                clean.append(text)

        return "\n".join(clean)[:1200] if clean else None
    except Exception as e:
        print(f"[search] DDG HTML search failed: {e}")
        return None


def web_search(query: str) -> str | None:
    """Use Tavily if configured, otherwise fall back to DuckDuckGo."""
    key = query.lower().strip()
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return result

    result = ddg_instant(query)
    if result:
        result = f"DuckDuckGo Instant Answer: {result}"
    else:
        tavily_result = tavily_search(query)
        if tavily_result:
            result = f"Tavily Search: {tavily_result}"
        else:
            result = ddg_search(query)

    if result:
        _cache[key] = (result, time.time())
    return result

def build_search_query(message: str) -> str:
    """Clean up the message into a good search query."""
    cleaned = re.sub(r'\b(corsbot|hey|please|bro|can you)\b', '', message.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()[:200]
