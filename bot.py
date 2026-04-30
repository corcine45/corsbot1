import discord
from groq import Groq
import sqlite3
import os
import time
import threading
import asyncio
import re
import requests
import random
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIG ---------------- #

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

client_ai = Groq(api_key=GROQ_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

executor = ThreadPoolExecutor(max_workers=4)

# ---------------- DB ---------------- #

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect("brain.db", check_same_thread=False)
        _local.cursor = _local.conn.cursor()

        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)

        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL,
                PRIMARY KEY (user_id, key)
            )
        """)

        # Migrate old schema (user_id, fact) -> (user_id, key, value, updated_at)
        cols = [row[1] for row in _local.cursor.execute("PRAGMA table_info(memory)").fetchall()]
        if "key" not in cols:
            _local.cursor.execute("ALTER TABLE memory RENAME TO memory_old")
            _local.cursor.execute("""
                CREATE TABLE memory (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    updated_at REAL,
                    PRIMARY KEY (user_id, key)
                )
            """)
            # Migrate old rows: fact was stored as "key=value"
            old_rows = _local.cursor.execute("SELECT user_id, fact FROM memory_old").fetchall()
            for user_id, fact in old_rows:
                if "=" in fact:
                    k, v = fact.split("=", 1)
                    _local.cursor.execute(
                        "INSERT OR IGNORE INTO memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                        (user_id, k.strip().lower(), v.strip(), time.time())
                    )
            _local.cursor.execute("DROP TABLE memory_old")

        _local.conn.commit()

    return _local.conn, _local.cursor

# ---------------- THREAD ---------------- #

def get_thread_id(user_id, guild_id=None, channel_id=None, is_dm=False):
    return f"dm:{user_id}" if is_dm else f"guild:{guild_id}:channel:{channel_id}"

# ---------------- HISTORY ---------------- #

def store_message(thread_id, role, content):
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (thread_id, role, content, time.time()),
    )
    conn.commit()

def get_history(thread_id, limit=10):
    _, cursor = get_db()
    cursor.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY timestamp DESC LIMIT ?",
        (thread_id, limit),
    )
    rows = cursor.fetchall()[::-1]
    return [{"role": r, "content": c} for r, c in rows]

# ---------------- 🧠 MEMORY ---------------- #

def extract_memory(user_id, message):
    # Skip extraction for short/casual messages — not worth the tokens
    if len(message.split()) < 5:
        return
    try:
        res = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract facts about the user from their message. "
                        "Reply in key=value format, one per line. "
                        "Use simple snake_case keys like: name, age, location, likes, dislikes, favorite_game, job, etc. "
                        "Only extract clear, stable personal facts explicitly stated by the user. "
                        "If the message contains no such facts, reply with exactly: NONE"
                    )
                },
                {"role": "user", "content": message}
            ],
            max_tokens=80,
        )

        output = res.choices[0].message.content.strip()

        if output.upper() == "NONE":
            return

        conn, cursor = get_db()

        for line in output.split("\n"):
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue

            # Upsert: insert or replace if key already exists for this user
            cursor.execute(
                """
                INSERT INTO memory (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (str(user_id), key, value, time.time()),
            )

        conn.commit()

    except Exception as e:
        print(f"[memory extract error] {e}")

def get_memory(user_id):
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value FROM memory WHERE user_id=? ORDER BY updated_at DESC LIMIT 20",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        return ""
    return "\n".join([f"{k}={v}" for k, v in rows])

def store_user_name(user_id, display_name):
    """Always keep an up-to-date display_name entry for a user."""
    conn, cursor = get_db()
    cursor.execute(
        """
        INSERT INTO memory (user_id, key, value, updated_at)
        VALUES (?, 'display_name', ?, ?)
        ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (str(user_id), display_name, time.time()),
    )
    conn.commit()

# ---------------- EMOTION ---------------- #

GIF_MAP = {
    "laugh": ["laughing", "funny reaction"],
    "happy": ["happy dance", "excited"],
    "sad": ["crying", "sad face"],
    "angry": ["angry face", "rage"],
    "shock": ["shocked", "mind blown"],
    "celebrate": ["party celebration", "confetti", "birthday cake"]
}

def ai_detect_emotion(text):
    try:
        res = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return one word: happy, sad, angry, laugh, shock, celebrate, none"},
                {"role": "user", "content": text}
            ],
            max_tokens=10,
        )
        return res.choices[0].message.content.strip().lower()
    except:
        return "none"

# ---------------- 🔥 GIPHY FIX ---------------- #

def search_gif(query):
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
        if not data:
            return None

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

        return None

    except:
        return None

# ---------------- AI ---------------- #

SYSTEM_PROMPT = (
    "You are Corsbot, a chill Discord bot made by Corcine. "
    "Stay in character as Corsbot — don't refer to yourself as an AI or language model, just be the bot. "
    "If anyone asks who made you, say Corcine made you. "
    "Be friendly, fun, and concise. Don't over-explain things. "
    "You remember things users tell you about themselves — their name, preferences, interests, etc. "
    "When you know something about the user, naturally bring it up when relevant. "
    "If a user tells you something personal, acknowledge that you'll remember it. "
    "You can discuss serious, historical, or controversial topics when asked. "
    "Do not avoid or brush off these topics — respond in an informative, neutral, and factual way. "
    "If a topic is sensitive like war, historical figures, or tragedies, just answer normally."
)

def ai_chat(history, memory, username=None):
    system = SYSTEM_PROMPT

    if username:
        system += f"\n\nYou are currently talking to {username}."

    if memory:
        system += (
            "\n\nKnown facts about this user:\n"
            + memory +
            "\nUse these when relevant."
        )

    messages = [{"role": "system", "content": system}] + history

    res = client_ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512,
    )

    return res.choices[0].message.content

# ---------------- SEND ---------------- #

async def send_reply(channel, text):
    limit = 2000
    while len(text) > limit:
        split = text.rfind("\n", 0, limit)
        if split == -1:
            split = limit
        await channel.send(text[:split])
        text = text[split:].lstrip("\n")
    if text:
        await channel.send(text)

# ---------------- BOT ---------------- #

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)

    should_reply = is_dm or (client.user in message.mentions)

    content = message.content.strip()
    if not should_reply or not content:
        return

    # Strip bot mention from content so it doesn't pollute history/memory
    content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
    if not content:
        return

    # Resolve other user mentions into display names and store facts about them
    mentioned_users = {}  # user_id -> display_name
    for user in message.mentions:
        if user == client.user:
            continue
        display = user.display_name
        content = content.replace(f"<@{user.id}>", f"@{display}").replace(f"<@!{user.id}>", f"@{display}")
        mentioned_users[user.id] = display

    thread_id = get_thread_id(
        message.author.id,
        message.guild.id if message.guild else None,
        message.channel.id,
        is_dm,
    )

    loop = asyncio.get_event_loop()

    # store user message
    await loop.run_in_executor(executor, store_message, thread_id, "user", content)

    # 🧠 update memory (blocking Groq call → executor)
    await loop.run_in_executor(executor, extract_memory, message.author.id, content)
    memory = await loop.run_in_executor(executor, get_memory, message.author.id)

    # Store facts about mentioned users too
    for uid, display in mentioned_users.items():
        # Always store their display name so the bot knows who they are
        await loop.run_in_executor(executor, store_user_name, uid, display)
        # Extract any facts said about them from the message
        await loop.run_in_executor(executor, extract_memory, uid, content)

    # fetch history
    history = await loop.run_in_executor(executor, get_history, thread_id)

    async with message.channel.typing():
        try:
            reply = await loop.run_in_executor(executor, ai_chat, history, memory, message.author.display_name)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                # Pull wait time from error message if available
                match = re.search(r"try again in ([^\.]+)", err)
                wait = match.group(1) if match else "a few minutes"
                await message.channel.send(f"⏳ I'm being rate limited, try again in {wait}.")
            else:
                await message.channel.send("Something went wrong, try again.")
                print(f"[ai_chat error] {e}")
            return

    await loop.run_in_executor(executor, store_message, thread_id, "assistant", reply)

    # ---------------- GIF ---------------- #

    emotion = await loop.run_in_executor(executor, ai_detect_emotion, (content + " " + reply)[:300])

    gif_url = None

    chance = {
        "laugh": 0.20,
        "celebrate": 0.15,
        "shock": 0.15,
        "happy": 0.10,
        "sad": 0.08,
        "angry": 0.08
    }

    if emotion in GIF_MAP and random.random() < chance.get(emotion, 0.15):
        gif_query = random.choice(GIF_MAP[emotion])
        gif_url = await loop.run_in_executor(executor, search_gif, gif_query)

    await send_reply(message.channel, reply)

    if gif_url:
        await message.channel.send(gif_url)
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", f"[GIF: {emotion}]")

# ---------------- RUN ---------------- #

client.run(DISCORD_TOKEN)