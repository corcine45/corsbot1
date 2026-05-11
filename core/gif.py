import aiohttp
import random
import time

from config import settings
from core.logger import get_logger

log = get_logger("corsbot.gif")

TENOR_API_KEY = settings.tenor_api_key

HEADERS = {"User-Agent": "Mozilla/5.0"}

# GIF results are stable — cache for 10 minutes per query
_gif_cache: dict[str, tuple[str, float]] = {}
_GIF_CACHE_TTL = 600
_GIF_CACHE_MAX = 200


def _gif_cache_get(query: str) -> str | None:
    entry = _gif_cache.get(query)
    if not entry:
        return None
    url, ts = entry
    if time.time() - ts > _GIF_CACHE_TTL:
        del _gif_cache[query]
        return None
    return url


def _gif_cache_set(query: str, url: str):
    if len(_gif_cache) >= _GIF_CACHE_MAX:
        oldest = next(iter(_gif_cache))
        del _gif_cache[oldest]
    _gif_cache[query] = (url, time.time())


async def search_gif_tenor(query: str) -> str | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://tenor.googleapis.com/v2/search",
                params={"q": query, "key": TENOR_API_KEY, "limit": 20, "contentfilter": "medium"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
        results = data.get("results", [])
        if not results:
            return None
        random.shuffle(results)
        for item in results:
            url = item.get("media_formats", {}).get("gif", {}).get("url")
            if url and url.startswith("https"):
                return url
    except Exception as e:
        log.warning("tenor_failed", query=query, error=str(e))
    return None


async def search_gif(query: str) -> str | None:
    """Search Giphy first, fall back to Tenor. Results cached per query."""
    cached = _gif_cache_get(query)
    if cached:
        return cached

    url = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.giphy.com/v1/gifs/search",
                params={"api_key": settings.giphy_api_key, "q": query, "limit": 25, "rating": "pg-13"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
        gifs = data.get("data", [])
        if gifs:
            random.shuffle(gifs)
            for gif in gifs[:5]:
                candidate = gif["images"]["downsized"]["url"]
                if candidate and candidate.startswith("https"):
                    url = candidate
                    break
    except Exception as e:
        log.warning("giphy_failed", query=query, error=str(e))

    if not url:
        url = await search_gif_tenor(query)

    if url:
        _gif_cache_set(query, url)

    return url
