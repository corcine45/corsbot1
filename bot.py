import discord
import asyncio
import re
import random
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# ---------------- ENV VALIDATION ---------------- #

_REQUIRED_ENV = {
    "DISCORD_TOKEN": "Discord bot token",
    "GROQ_API_KEY":  "Groq API key",
    "GIPHY_API_KEY": "Giphy API key",
}

_missing = [f"{var} ({desc})" for var, desc in _REQUIRED_ENV.items() if not os.getenv(var)]
if _missing:
    print("❌ Missing required environment variables:")
    for m in _missing:
        print(f"   • {m}")
    print("Set them in your .env file or as system environment variables.")
    raise SystemExit(1)

print("✅ All environment variables loaded.")

from core.db import get_db, get_thread_id, store_message, get_history
from core.ai import (
    ai_chat, detect_mood, set_mood, get_current_mood,
    MOOD_PROMPTS, FALLBACK_RESPONSES, is_prompt_injection
)
from core.memory import (
    extract_memory, get_memory, get_memory_with_keys, store_user_name,
    extract_relationships, get_relationships, should_extract
)
from core.emotion import pick_gif_for_message
from core.search import needs_web_search, web_search, build_search_query
from core.feedback import (
    store_last_reply, get_last_reply, store_feedback,
    apply_good_rating, apply_bad_rating, get_feedback_stats
)

# ---------------- CONFIG ---------------- #

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

executor = ThreadPoolExecutor(max_workers=4)

# ---------------- QUICK REPLIES ---------------- #

QUICK_REPLIES = {
    frozenset(["hi", "hey", "hello", "sup", "yo", "hiya"]): ["hey!", "yo!", "sup", "heyyy", "what's good"],
    frozenset(["lol", "lmao", "lmfao", "haha", "hahaha", "😂", "💀"]): ["💀", "lmaooo", "bro 😭", "nah fr 💀"],
    frozenset(["ok", "okay", "k", "kk", "alright", "aight"]): ["aight", "ok", "cool", "bet"],
    frozenset(["thanks", "thank you", "ty", "thx"]): ["np!", "anytime", "of course", "👍"],
    frozenset(["bye", "cya", "see ya", "later", "gtg"]): ["later!", "cya", "peace ✌️", "see ya"],
}

def get_quick_reply(text: str):
    normalized = text.lower().strip().rstrip("!?.")
    for patterns, replies in QUICK_REPLIES.items():
        if normalized in patterns:
            return random.choice(replies)
    return None

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

def resolve_mentions_in_reply(reply, guild):
    """Convert @name text in AI reply to real Discord mentions <@id>."""
    if not guild:
        return reply
    def replace_mention(match):
        name = match.group(1).lower()
        for member in guild.members:
            if member.display_name.lower() == name or member.name.lower() == name:
                return f"<@{member.id}>"
        return match.group(0)
    return re.sub(r"@([\w\s]+)", replace_mention, reply)

# ---------------- COMMANDS ---------------- #

VALID_MOODS = list(MOOD_PROMPTS.keys())

async def cmd_memory(message, user_id):
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value, memory_type, reinforcement, updated_at FROM memory WHERE user_id=? ORDER BY memory_type, key",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        await message.channel.send(f"<@{user_id}> I don't know anything about you yet.")
        return
    import time
    lines = ["📋 **What I know about you:**"]
    for key, value, mtype, reinforcement, updated_at in rows:
        age_days = (time.time() - updated_at) / 86400
        icon = {"identity": "🔒", "preference": "⭐", "temporary": "⏳"}.get(mtype or "preference", "•")
        lines.append(f"{icon} `{key}` = {value}  _(seen {reinforcement}x, {age_days:.0f}d ago)_")
    await send_reply(message.channel, "\n".join(lines))

async def cmd_forget(message, user_id, args):
    if not args:
        await message.channel.send("Usage: `!forget <key>` — e.g. `!forget likes`")
        return
    key = args[0].lower()
    conn, cursor = get_db()
    cursor.execute("DELETE FROM memory WHERE user_id=? AND key=?", (str(user_id), key))
    conn.commit()
    if cursor.rowcount:
        await message.channel.send(f"<@{user_id}> Forgot `{key}`. ✅")
    else:
        await message.channel.send(f"<@{user_id}> Nothing stored under `{key}`.")

async def cmd_forget_all(message, user_id):
    conn, cursor = get_db()
    cursor.execute("DELETE FROM memory WHERE user_id=?", (str(user_id),))
    conn.commit()
    await message.channel.send(f"<@{user_id}> Cleared all your memories. Fresh start. 🧹")

async def cmd_personality(message, user_id, args):
    if not args or args[0].lower() not in VALID_MOODS:
        await message.channel.send(f"Available moods: `{'`, `'.join(VALID_MOODS)}`")
        return
    mood = args[0].lower()
    set_mood(str(user_id), mood)
    await message.channel.send(f"<@{user_id}> Switched to **{mood}** mode. 🎭")

async def cmd_stats(message, thread_id, user_id):
    import time
    _, cursor = get_db()
    cursor.execute("SELECT COUNT(*) FROM messages WHERE thread_id=?", (thread_id,))
    msg_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*), memory_type FROM memory WHERE user_id=? GROUP BY memory_type", (str(user_id),))
    mem_rows = cursor.fetchall()
    current_mood = get_current_mood(str(user_id))
    lines = [
        "📊 **Stats**",
        f"💬 Messages in this thread: **{msg_count}**",
        f"🎭 Current mood: **{current_mood}**",
        "🧠 Memory:",
    ]
    for count, mtype in mem_rows:
        icon = {"identity": "🔒", "preference": "⭐", "temporary": "⏳"}.get(mtype or "preference", "•")
        lines.append(f"  {icon} {mtype or 'preference'}: **{count}** facts")
    await message.channel.send("\n".join(lines))

async def cmd_relationships(message, user_id):
    import time
    _, cursor = get_db()
    cursor.execute(
        "SELECT related_name, relation, context, strength, updated_at FROM relationships WHERE user_id=? ORDER BY strength DESC",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        await message.channel.send(f"<@{user_id}> I don't know anyone in your life yet — tell me about your friends!")
        return
    lines = ["👥 **People I know about:**"]
    for name, relation, context, strength, updated_at in rows:
        age_days = (time.time() - updated_at) / 86400
        line = f"• **{name}** ({relation})"
        if context:
            line += f" — {context}"
        line += f"  _(mentioned {strength}x, {age_days:.0f}d ago)_"
        lines.append(line)
    await send_reply(message.channel, "\n".join(lines))

async def cmd_help(message):
    lines = [
        "🤖 **Corsbot Commands**",
        "`!memory` — see everything I know about you",
        "`!forget <key>` — delete a specific memory",
        "`!forgetall` — wipe all your memories",
        f"`!personality <mood>` — set my mood (`{'`, `'.join(VALID_MOODS)}`)",
        "`!stats` — conversation and memory stats",
        "`!relationships` — see who I know about in your life",
        "`!rate good` / `!rate bad` — rate my last reply",
        "`!ratings` — see your rating history",
        "`!help` — show this list",
    ]
    await message.channel.send("\n".join(lines))

async def cmd_rate(message, user_id, thread_id, args):
    if not args or args[0].lower() not in ("good", "bad"):
        await message.channel.send("Usage: `!rate good` or `!rate bad`")
        return

    rating = args[0].lower()
    entry = get_last_reply(str(user_id))

    if not entry:
        await message.channel.send(f"<@{user_id}> No recent reply to rate — ratings expire after 5 minutes.")
        return

    store_feedback(str(user_id), thread_id, entry["reply"], rating, entry["mood"])

    if rating == "good":
        apply_good_rating(str(user_id), entry["mood"], entry["memory_keys"])
        await message.channel.send(f"<@{user_id}> glad you liked it 🙏")
    else:
        apply_bad_rating(str(user_id), entry["mood"])
        await message.channel.send(f"<@{user_id}> noted, i'll do better 🫡")

async def cmd_ratings(message, user_id):
    stats = get_feedback_stats(str(user_id))
    total = stats["good"] + stats["bad"]
    if total == 0:
        await message.channel.send(f"<@{user_id}> No ratings yet — use `!rate good` or `!rate bad` after my replies.")
        return
    pct = int((stats['good'] / total) * 100) if total else 0
    await message.channel.send(
        f"<@{user_id}> 📊 Your ratings: ✅ {stats['good']} good · ❌ {stats['bad']} bad · {pct}% satisfaction"
    )

async def handle_command(message, content, thread_id):
    parts = content.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]
    uid = message.author.id
    if cmd == "!memory":          await cmd_memory(message, uid)
    elif cmd == "!forget":        await cmd_forget(message, uid, args)
    elif cmd == "!forgetall":     await cmd_forget_all(message, uid)
    elif cmd == "!personality":   await cmd_personality(message, uid, args)
    elif cmd == "!stats":         await cmd_stats(message, thread_id, uid)
    elif cmd == "!relationships": await cmd_relationships(message, uid)
    elif cmd == "!rate":          await cmd_rate(message, uid, thread_id, args)
    elif cmd == "!ratings":       await cmd_ratings(message, uid)
    elif cmd == "!help":          await cmd_help(message)
    else:                         return False
    return True

# ---------------- BOT EVENTS ---------------- #

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

    # Strip bot mention
    content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
    if not content:
        return

    # Prompt injection guard
    if is_prompt_injection(content):
        await message.channel.send(f"<@{message.author.id}> nice try 💀")
        return

    # Resolve other user mentions
    mentioned_users = {}
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

    await loop.run_in_executor(executor, store_message, thread_id, "user", content)

    # Commands
    if content.startswith("!"):
        handled = await handle_command(message, content, thread_id)
        if handled:
            return

    # Quick replies
    quick = get_quick_reply(content)
    if quick:
        await message.channel.send(f"<@{message.author.id}> {quick}")
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", quick)
        return

    # Memory + relationships (batched)
    uid_str = str(message.author.id)
    if should_extract(uid_str):
        await loop.run_in_executor(executor, extract_memory, message.author.id, content)
        await loop.run_in_executor(executor, extract_relationships, message.author.id, content)

    memory, active_keys = await loop.run_in_executor(executor, get_memory_with_keys, message.author.id, content)
    relationships = await loop.run_in_executor(executor, get_relationships, message.author.id)

    # Web search if message needs real-time info
    web_context = ""
    if needs_web_search(content):
        query = build_search_query(content)
        web_context = await loop.run_in_executor(executor, web_search, query) or ""
        if web_context:
            print(f"[search] fetched context for: {query}")

    # Mentioned users
    for uid, display in mentioned_users.items():
        await loop.run_in_executor(executor, store_user_name, uid, display)
        if should_extract(str(uid)):
            await loop.run_in_executor(executor, extract_memory, uid, content)

    history = await loop.run_in_executor(executor, get_history, thread_id)
    mood = detect_mood(uid_str, history)

    async with message.channel.typing():
        try:
            reply = await loop.run_in_executor(
                executor, ai_chat, history, memory,
                message.author.display_name, mood, relationships, web_context
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in ([^\.]+)", err)
                wait = match.group(1) if match else "a few minutes"
                await message.channel.send(f"<@{message.author.id}> ⏳ rate limited, try again in {wait}.")
            else:
                await message.channel.send(f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}")
                print(f"[on_message error] {e}")
            return

    await loop.run_in_executor(executor, store_message, thread_id, "assistant", reply)

    # Store last reply for !rate
    store_last_reply(str(message.author.id), reply, mood, active_keys)
    reply = resolve_mentions_in_reply(reply, message.guild)

    gif_url, emotion = await loop.run_in_executor(executor, pick_gif_for_message, content, reply)

    await send_reply(message.channel, f"<@{message.author.id}> {reply}")

    if gif_url:
        await message.channel.send(gif_url)
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", f"[GIF: {emotion}]")

# ---------------- RUN ---------------- #

client.run(DISCORD_TOKEN)
