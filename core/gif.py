import aiohttp
import random
import time
from collections import deque

from config import settings
from core.logger import get_logger

log = get_logger("corsbot.gif")

TENOR_API_KEY = settings.tenor_api_key
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Cache a pool of URLs per query so repeated requests get different GIFs
_gif_pool_cache: dict[str, tuple[deque, float]] = {}
_GIF_CACHE_TTL = 600   # 10 minutes
_GIF_CACHE_MAX = 200
_POOL_SIZE = 10        # keep up to 10 URLs per query


def _pool_get(query: str) -> str | None:
    """Get the next URL from the pool, rotating so each call returns a different one."""
    entry = _gif_pool_cache.get(query)
    if not entry:
        return None
    pool, ts = entry
    if time.time() - ts > _GIF_CACHE_TTL:
        del _gif_pool_cache[query]
        return None
    if not pool:
        return None
    url = pool[0]
    pool.rotate(-1)  # move used URL to the back
    return url


def _pool_set(query: str, urls: list[str]):
    if len(_gif_pool_cache) >= _GIF_CACHE_MAX:
        oldest = next(iter(_gif_pool_cache))
        del _gif_pool_cache[oldest]
    random.shuffle(urls)
    _gif_pool_cache[query] = (deque(urls[:_POOL_SIZE]), time.time())


async def search_gif_tenor(query: str) -> list[str]:
    """Fetch up to _POOL_SIZE GIF URLs from Tenor."""
    urls = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://tenor.googleapis.com/v2/search",
                params={"q": query, "key": TENOR_API_KEY, "limit": 20, "contentfilter": "medium"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
        results = data.get("results", [])
        random.shuffle(results)
        for item in results:
            url = item.get("media_formats", {}).get("gif", {}).get("url")
            if url and url.startswith("https"):
                urls.append(url)
            if len(urls) >= _POOL_SIZE:
                break
    except Exception as e:
        log.warning("tenor_failed", query=query, error=str(e))
    return urls


async def search_gif(query: str) -> str | None:
    """
    Return a GIF URL for the query.
    Builds a pool of results on first call, then rotates through them
    so repeated calls for the same query return different GIFs.
    """
    # Try pool cache first
    cached = _pool_get(query)
    if cached:
        return cached

    # Fetch fresh results
    urls = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.giphy.com/v1/gifs/search",
                params={"api_key": settings.giphy_api_key, "q": query, "limit": 25, "rating": "pg-13"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
        gifs = data.get("data", [])
        random.shuffle(gifs)
        for gif in gifs:
            url = gif["images"]["downsized"]["url"]
            if url and url.startswith("https"):
                urls.append(url)
            if len(urls) >= _POOL_SIZE:
                break
    except Exception as e:
        log.warning("giphy_failed", query=query, error=str(e))

    if not urls:
        urls = await search_gif_tenor(query)

    if not urls:
        return None

    _pool_set(query, urls)
    return _pool_get(query)
