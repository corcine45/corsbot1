import aiohttp
import os
import re
import time

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
    "news", "latest", "today's", "right now", "live score", "scores",
    "standings", "match result", "breaking", "breaking news",
    "what's happening", "whats happening",
    "who won", "game score", "match score", "final score",
    "weather", "temperature", "forecast",
    "stock price", "crypto price", "bitcoin price", "ethereum price", "exchange rate",
    "patch notes", "new update", "new patch", "new season", "new episode",
    "just dropped", "just announced", "just released", "just launched",
    "release date", "when does", "when is",
    "what happened to", "what's the latest", "whats the latest",
    "is it out", "is it available", "is it live",
    "who is", "who's", "whos",
}

def needs_web_search(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in REALTIME_TRIGGERS)


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
        snippets = [t["Text"] for t in topics[:3] if isinstance(t, dict) and t.get("Text")]
        if snippets:
            return " | ".join(snippets)[:600]
    except Exception as e:
        print(f"[search] DDG instant failed: {e}")
    return None


async def tavily_search(session: aiohttp.ClientSession, query: str, max_results: int = 3) -> str | None:
    if not settings.tavily_api_key:
        return None
    try:
        async with session.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}", "Content-Type": "application/json"},
            json={"query": query, "max_results": max_results, "include_answer": False, "search_depth": "basic"},
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
            snippet = re.sub(r"\s+", " ", item.get("content", "") or item.get("raw_content", "")).strip()
            if title and snippet:
                clean.append(f"{title} — {snippet} [source: {url}]")
            elif snippet:
                clean.append(f"{snippet} [source: {url}]")
        return "\n".join(clean)[:1200] if clean else None
    except Exception as e:
        print(f"[search] Tavily failed: {e}")
    return None


async def ddg_search(session: aiohttp.ClientSession, query: str, max_results: int = 3) -> str | None:
    try:
        async with session.post(
            DDG_SEARCH_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            html = await resp.text()
        results = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        clean = []
        for idx, (link, raw_title) in enumerate(results[:max_results]):
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", snippets[idx])).strip() if idx < len(snippets) else ""
            if title or snippet:
                clean.append(f"{title} — {snippet} [source: {link}]" if snippet else f"{title} [source: {link}]")
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
    cleaned = re.sub(r'\b(corsbot|hey|please|bro|can you)\b', '', message.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()[:200]
