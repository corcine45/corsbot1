import discord
from discord import app_commands
import asyncio
import re
import random
import time
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
    ai_chat, get_mood, set_mood, get_current_mood, reset_mood,
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
VALID_MOODS = list(MOOD_PROMPTS.keys())

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

class CorsBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        print("✅ Slash commands synced.")

client = CorsBot()
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

# ---------------- HELPERS ---------------- #

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

async def send_interaction(interaction: discord.Interaction, text: str, ephemeral: bool = True):
    """Send a slash command response, splitting if needed."""
    if len(text) <= 2000:
        await interaction.response.send_message(text, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(text[:2000], ephemeral=ephemeral)
        rest = text[2000:]
        while rest:
            await interaction.followup.send(rest[:2000], ephemeral=ephemeral)
            rest = rest[2000:]

def resolve_mentions_in_reply(reply, guild):
    if not guild:
        return reply
    def replace_mention(match):
        name = match.group(1).lower()
        for member in guild.members:
            if member.display_name.lower() == name or member.name.lower() == name:
                return f"<@{member.id}>"
        return match.group(0)
    return re.sub(r"@([\w\s]+)", replace_mention, reply)

# ---------------- SLASH COMMANDS ---------------- #

@client.tree.command(name="memory", description="See everything Corsbot knows about you")
async def slash_memory(interaction: discord.Interaction):
    user_id = interaction.user.id
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value, memory_type, reinforcement, updated_at FROM memory WHERE user_id=? ORDER BY memory_type, key",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        await interaction.response.send_message("I don't know anything about you yet.", ephemeral=True)
        return
    lines = ["📋 **What I know about you:**"]
    for key, value, mtype, reinforcement, updated_at in rows:
        age_days = (time.time() - updated_at) / 86400
        icon = {"identity": "🔒", "preference": "⭐", "temporary": "⏳"}.get(mtype or "preference", "•")
        lines.append(f"{icon} `{key}` = {value}  _(seen {reinforcement}x, {age_days:.0f}d ago)_")
    await send_interaction(interaction, "\n".join(lines))


@client.tree.command(name="forget", description="Delete a specific memory by key")
@app_commands.describe(key="The memory key to delete (use /memory to see keys)")
async def slash_forget(interaction: discord.Interaction, key: str):
    user_id = interaction.user.id
    conn, cursor = get_db()
    cursor.execute("DELETE FROM memory WHERE user_id=? AND key=?", (str(user_id), key.lower()))
    conn.commit()
    if cursor.rowcount:
        await interaction.response.send_message(f"Forgot `{key}`. ✅", ephemeral=True)
    else:
        await interaction.response.send_message(f"Nothing stored under `{key}`.", ephemeral=True)


@client.tree.command(name="forgetall", description="Wipe everything Corsbot knows about you")
async def slash_forget_all(interaction: discord.Interaction):
    user_id = interaction.user.id
    conn, cursor = get_db()
    cursor.execute("DELETE FROM memory WHERE user_id=?", (str(user_id),))
    conn.commit()
    await interaction.response.send_message("Cleared all your memories. Fresh start. 🧹", ephemeral=True)


@client.tree.command(name="personality", description="Set Corsbot's mood when talking to you")
@app_commands.describe(mood="Choose a mood")
@app_commands.choices(mood=[
    app_commands.Choice(name=m, value=m) for m in VALID_MOODS
])
async def slash_personality(interaction: discord.Interaction, mood: str):
    set_mood(str(interaction.user.id), mood)
    await interaction.response.send_message(f"Switched to **{mood}** mode. 🎭", ephemeral=True)


@client.tree.command(name="reset", description="Reset personality to chill")
async def slash_reset(interaction: discord.Interaction):
    reset_mood(str(interaction.user.id))
    await interaction.response.send_message("Reset to **chill** mode. ✨", ephemeral=True)


@client.tree.command(name="stats", description="Show conversation and memory stats")
async def slash_stats(interaction: discord.Interaction):
    user_id = interaction.user.id
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    thread_id = get_thread_id(
        user_id,
        interaction.guild_id if interaction.guild else None,
        interaction.channel_id,
        is_dm,
    )
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
    await send_interaction(interaction, "\n".join(lines))


@client.tree.command(name="relationships", description="See people Corsbot knows about in your life")
async def slash_relationships(interaction: discord.Interaction):
    user_id = interaction.user.id
    _, cursor = get_db()
    cursor.execute(
        "SELECT related_name, relation, context, strength, updated_at FROM relationships WHERE user_id=? ORDER BY strength DESC",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        await interaction.response.send_message(
            "I don't know anyone in your life yet — tell me about your friends!", ephemeral=True
        )
        return
    lines = ["👥 **People I know about:**"]
    for name, relation, context, strength, updated_at in rows:
        age_days = (time.time() - updated_at) / 86400
        line = f"• **{name}** ({relation})"
        if context:
            line += f" — {context}"
        line += f"  _(mentioned {strength}x, {age_days:.0f}d ago)_"
        lines.append(line)
    await send_interaction(interaction, "\n".join(lines))


@client.tree.command(name="rate", description="Rate Corsbot's last reply")
@app_commands.describe(rating="Was the reply good or bad?")
@app_commands.choices(rating=[
    app_commands.Choice(name="👍 Good", value="good"),
    app_commands.Choice(name="👎 Bad",  value="bad"),
])
async def slash_rate(interaction: discord.Interaction, rating: str):
    user_id = interaction.user.id
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    thread_id = get_thread_id(
        user_id,
        interaction.guild_id if interaction.guild else None,
        interaction.channel_id,
        is_dm,
    )
    entry = get_last_reply(str(user_id))
    if not entry:
        await interaction.response.send_message(
            "No recent reply to rate — ratings expire after 5 minutes.", ephemeral=True
        )
        return
    store_feedback(str(user_id), thread_id, entry["reply"], rating, entry["mood"])
    if rating == "good":
        apply_good_rating(str(user_id), entry["mood"], entry["memory_keys"])
        await interaction.response.send_message("glad you liked it 🙏", ephemeral=True)
    else:
        apply_bad_rating(str(user_id), entry["mood"])
        await interaction.response.send_message("noted, i'll do better 🫡", ephemeral=True)


@client.tree.command(name="ratings", description="See your rating history for Corsbot")
async def slash_ratings(interaction: discord.Interaction):
    stats = get_feedback_stats(str(interaction.user.id))
    total = stats["good"] + stats["bad"]
    if total == 0:
        await interaction.response.send_message(
            "No ratings yet — use `/rate` after my replies.", ephemeral=True
        )
        return
    pct = int((stats["good"] / total) * 100)
    await interaction.response.send_message(
        f"📊 Your ratings: ✅ {stats['good']} good · ❌ {stats['bad']} bad · {pct}% satisfaction",
        ephemeral=True
    )


@client.tree.command(name="help", description="Show all Corsbot commands")
async def slash_help(interaction: discord.Interaction):
    lines = [
        "🤖 **Corsbot Commands**",
        "`/memory` — see everything I know about you",
        "`/forget <key>` — delete a specific memory",
        "`/forgetall` — wipe all your memories",
        "`/personality <mood>` — set my mood",
        "`/stats` — conversation and memory stats",
        "`/relationships` — see who I know about in your life",
        "`/rate` — rate my last reply",
        "`/ratings` — see your rating history",
        "`/help` — show this list",
    ]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

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

    # Prefix message with sender name so AI never confuses sender with mentioned users
    attributed_content = f"[{message.author.display_name}]: {content}"
    await loop.run_in_executor(executor, store_message, thread_id, "user", attributed_content)

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
        await loop.run_in_executor(executor, extract_relationships, message.author.id, attributed_content)

    memory, active_keys = await loop.run_in_executor(executor, get_memory_with_keys, message.author.id, content)
    relationships = await loop.run_in_executor(executor, get_relationships, message.author.id)

    # If impersonating a mentioned user, fetch their memory and inject it
    impersonation_context = ""
    impersonate_keywords = ("pretend", "act as", "be ", "impersonate", "roleplay as", "talk like", "speak as")
    lower_content = content.lower()
    if any(kw in lower_content for kw in impersonate_keywords) and mentioned_users:
        target_id = next(iter(mentioned_users))
        target_name = mentioned_users[target_id]
        target_memory = await loop.run_in_executor(executor, get_memory, target_id)
        if target_memory:
            impersonation_context = f"Facts about {target_name} to help you impersonate them:\n{target_memory}"

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
    mood = get_mood(uid_str)

    async with message.channel.typing():
        try:
            reply = await loop.run_in_executor(
                executor, ai_chat, history, memory,
                message.author.display_name, mood, relationships, web_context, impersonation_context
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
    store_last_reply(str(message.author.id), reply, mood, active_keys)
    reply = resolve_mentions_in_reply(reply, message.guild)

    gif_url, emotion = await loop.run_in_executor(executor, pick_gif_for_message, content, reply)

    await send_reply(message.channel, f"<@{message.author.id}> {reply}")

    if gif_url:
        await message.channel.send(gif_url)
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", f"[GIF: {emotion}]")

# ---------------- RUN ---------------- #

client.run(DISCORD_TOKEN)
