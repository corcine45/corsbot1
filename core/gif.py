import requests
import random
import os

GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

def search_gif_tenor(query: str):
    """Fallback GIF search using Tenor."""
    try:
        res = requests.get(
            "https://tenor.googleapis.com/v2/search",
            params={
                "q": query,
                "key": "AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCyk",
                "limit": 20,
                "contentfilter": "medium",
            },
            timeout=5
        )
        results = res.json().get("results", [])
        if not results:
            return None
        random.shuffle(results)
        for item in results:
            url = item.get("media_formats", {}).get("gif", {}).get("url")
            if url and url.startswith("https"):
                return url
        return None
    except:
        return None

def search_gif(query: str):
    """Search Giphy first, fall back to Tenor."""
    try:
        res = requests.get(
            "https://api.giphy.com/v1/gifs/search",
            params={
                "api_key": GIPHY_API_KEY,
                "q": query,
                "limit": 25,
                "rating": "pg-13"
            },
            timeout=5
        )
        data = res.json().get("data", [])
        if data:
            random.shuffle(data)
            for gif in data:
                url = gif["images"]["downsized"]["url"]
                if url and url.startswith("https"):
                    try:
                        head = requests.head(url, timeout=3)
                        if head.status_code == 200:
                            return url
                    except:
                        continue
    except:
        pass
    return search_gif_tenor(query)
