import requests
import re
import time

# DuckDuckGo instant answer + HTML search — no API key required
DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"
DDG_INSTANT_URL = "https://api.duckduckgo.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

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
    return any(trigger in lower for trigger in REALTIME_TRIGGERS)

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

def ddg_search(query: str, max_results: int = 3) -> str | None:
    """Scrape DuckDuckGo HTML search for snippets."""
    try:
        res = requests.post(
            DDG_SEARCH_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=8,
        )
        html = res.text

        # Extract result snippets using regex (avoids needing BeautifulSoup)
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        # Clean HTML tags
        clean = []
        for s in snippets[:max_results]:
            s = re.sub(r"<[^>]+>", "", s).strip()
            s = re.sub(r"\s+", " ", s)
            if s:
                clean.append(s)

        return " | ".join(clean)[:800] if clean else None
    except Exception as e:
        print(f"[search] DDG HTML search failed: {e}")
        return None

def web_search(query: str) -> str | None:
    """Try instant answer first, fall back to HTML search."""
    result = ddg_instant(query)
    if result:
        return result
    return ddg_search(query)

def build_search_query(message: str) -> str:
    """Clean up the message into a good search query."""
    # Remove filler words
    filler = {"can you", "do you know", "what is", "tell me about", "hey", "corsbot", "please", "bro"}
    words = message.lower().split()
    filtered = [w for w in words if w not in filler]
    return " ".join(filtered)[:200]
