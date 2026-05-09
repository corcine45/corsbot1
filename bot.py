import discord
from discord import app_commands
import aiohttp
import asyncio
import base64
import hashlib
import logging
import re
import random
import time
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("corsbot")

# ---------------- ENV VALIDATION ---------------- #

_REQUIRED_ENV = {
    "DISCORD_TOKEN": "Discord bot token",
    "GROQ_API_KEY":  "Groq API key",
    "GIPHY_API_KEY": "Giphy API key",
}

_missing = [f"{var} ({desc})" for var, desc in _REQUIRED_ENV.items() if not os.getenv(var)]
if _missing:
    for m in _missing:
        log.error(f"Missing env var: {m}")
    raise SystemExit(1)

log.info("All environment variables loaded.")

from core.db import get_db, get_thread_id, store_message, get_history
from core.ai import ai_chat, FALLBACK_RESPONSES, is_prompt_injection, is_semantic_jailbreak, sanitize_retrieved_content, groq_call
from core.memory import (
    extract_memory, store_user_name,
    extract_relationships, should_extract,
    check_and_delete_denied_facts, delete_denied_fact,
    update_reflection, should_update_reflection,
)
from core.emotion import pick_gif_for_message
from core.search import needs_web_search, web_search, build_search_query
from core.session import add_message, should_refresh, get_recent_messages, get_state_prompt, analyze_state, set_state
from core.feedback import (
    store_last_reply, get_last_reply, store_feedback,
    apply_good_rating, apply_bad_rating, get_feedback_stats,
    get_feedback_context,
)

# ---------------- DENIAL CONFIRMATION ---------------- #

class DenialConfirmView(discord.ui.View):
    def __init__(self, user_id: int, facts: list):
        super().__init__(timeout=60)
        self.user_id = user_id
        self.facts = facts  # list of (key, value)

    @discord.ui.button(label="Yeah delete it", style=discord.ButtonStyle.red)
    async def confirm_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not for you.", ephemeral=True)
            return
        loop = asyncio.get_running_loop()
        for key, value in self.facts:
            await loop.run_in_executor(executor, delete_denied_fact, self.user_id, key)
        deleted = ", ".join(f"`{v}`" for _, v in self.facts)
        await interaction.response.send_message(f"Gone. Won't remember {deleted} about you anymore.", ephemeral=False)
        self.stop()

    @discord.ui.button(label="Nah keep it", style=discord.ButtonStyle.grey)
    async def cancel_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not for you.", ephemeral=True)
            return
        await interaction.response.send_message("Aight, keeping it.", ephemeral=False)
        self.stop()

# ---------------- CONFIG ---------------- #

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HISTORY_LIMIT = 30
COOLDOWN_SECONDS = 3
RESPONSE_CACHE_TTL = 300
_RESPONSE_CACHE: dict[str, tuple[str, float]] = {}
_user_cooldowns: dict[int, float] = {}

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
    frozenset([
        "hi", "hey", "hello", "sup", "yo", "hiya", "heya", "wassup", "wsp",
        "wazzup", "howdy", "ello", "helo", "heyy", "heyyyy", "yoo", "yooo",
        "oi", "ay", "ayy", "ayyy", "what's up", "whats up", "wuts up",
    ]): ["hey!", "yo!", "sup", "heyyy", "what's good", "ayy", "yoo", "what's good bro"],

    frozenset([
        "lol", "lmao", "lmfao", "haha", "hahaha", "hahahaha", "😂", "💀",
        "lmaoo", "lmaooo", "lmaoooo", "hahah", "hehe", "hihi", "xd", "XD",
        "💀💀", "😭😭", "dead", "im dead", "i'm dead", "bruh", "bruhhh",
        "lol lol", "lolol", "lololol", "kekw", "kek",
    ]): ["💀", "lmaooo", "bro 😭", "nah fr 💀", "bro 💀💀", "dead 😭", "kekw"],

    frozenset([
        "ok", "okay", "k", "kk", "kkk", "alright", "aight", "aite", "ight",
        "bet", "gotcha", "got it", "understood", "noted", "copy", "roger",
        "sure", "yep", "yup", "yeah", "ya", "ye", "yea", "mhm", "mmk",
    ]): ["aight", "ok", "cool", "bet", "gotcha", "noted", "yep"],

    frozenset([
        "thanks", "thank you", "ty", "thx", "thnx", "thank u", "thankyou",
        "tysm", "tyvm", "thanks a lot", "much appreciated", "appreciate it",
        "appreciate that", "cheers", "gracias", "salamat",
    ]): ["np!", "anytime", "of course", "👍", "always", "no worries", "salamat din"],

    frozenset([
        "bye", "cya", "see ya", "later", "gtg", "gotta go", "peace",
        "goodbye", "good bye", "byebye", "bye bye", "ttyl", "ttys",
        "take care", "tc", "laters", "adios", "ciao", "see you",
        "see u", "catch you later", "catch u later", "imma go",
    ]): ["later!", "cya", "peace ✌️", "see ya", "take care", "adios", "ttyl"],

    frozenset([
        "gm", "good morning", "morning", "mornin", "rise and shine",
    ]): ["gm!", "morning 🌅", "rise and grind", "gm gm"],

    frozenset([
        "gn", "good night", "night", "nite", "goodnight", "sleep well",
        "going to sleep", "gonna sleep", "imma sleep",
    ]): ["gn!", "sleep well 🌙", "night night", "rest up"],

    frozenset([
        "fr", "fr fr", "facts", "real", "no cap", "nocap", "deadass",
        "on god", "ong", "on gang", "frfr",
    ]): ["fr fr", "no cap", "facts", "deadass", "ong"],

    frozenset([
        "nah", "nope", "no", "nah bro", "nah man", "hell nah", "hell no",
    ]): ["nah?", "aight then", "ok ok", "fair enough"],

    frozenset([
        "gg", "good game", "ggs",
    ]): ["gg!", "ggs", "well played", "gg ez"],

    frozenset([
        "pog", "poggers", "lets go", "let's go", "lesgo", "letsgo", "w",
        "big w", "dub", "we won", "we cooked",
    ]): ["LETS GOOO 🔥", "W", "poggers", "big W", "we cooked fr"],
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

async def extract_attachment_text(attachment: discord.Attachment) -> str:
    """Use Groq vision to describe/read the image instead of OCR."""
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        return ""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                if response.status != 200:
                    print(f"[vision] failed to download image: {response.status}")
                    return ""
                data = await response.read()

        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{attachment.content_type};base64,{b64}"

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: groq_call(
                "meta-llama/llama-4-scout-17b-16e-instruct",
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": "Describe this image concisely. If it contains text, include it. If it's a meme, explain it.",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
        )
        print(f"[vision] result: {result[:80] if result else 'empty'}")
        return f"[Image: {result}]" if result else ""
    except Exception as e:
        print(f"[vision] failed: {e}")
        return ""

MEMORY_INJECTION_TRIGGERS = (
    "remember",
    "do you remember",
    "do you know",
    "what do you know",
    "what do you remember",
    "tell me about",
    "what can you tell me about",
    "what do you know about",
    "who am i",
    "what's my",
    "what is my",
    "do you have memory",
    "do you recall",
    "recall",
    "remind me",
    "about me",
)

def is_explicit_memory_request(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in MEMORY_INJECTION_TRIGGERS)


def build_response_cache_key(thread_id, content, memory, relationships, web_context, feedback_context):
    payload = "\n".join([
        thread_id,
        content.strip().lower(),
        memory or "",
        relationships or "",
        web_context or "",
        feedback_context or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_response(key):
    entry = _RESPONSE_CACHE.get(key)
    if not entry:
        return None
    reply, ts = entry
    if time.time() - ts > RESPONSE_CACHE_TTL:
        _RESPONSE_CACHE.pop(key, None)
        return None
    return reply


def set_cached_response(key, reply):
    _RESPONSE_CACHE[key] = (reply, time.time())

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
    cursor.execute("DELETE FROM relationships WHERE user_id=?", (str(user_id),))
    conn.commit()
    await interaction.response.send_message("Cleared all your memories and relationships. Fresh start. 🧹", ephemeral=True)


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
    lines = [
        "📊 **Stats**",
        f"💬 Messages in this thread: **{msg_count}**",
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
    store_feedback(str(user_id), thread_id, entry["reply"], rating, entry["mood"], interaction.guild_id if interaction.guild else None)
    if rating == "good":
        apply_good_rating(str(user_id), entry["mood"], entry["memory_keys"])
        await interaction.response.send_message("glad you liked it 🙏", ephemeral=True)
    else:
        apply_bad_rating(str(user_id), entry["mood"])
        await interaction.response.send_message("noted, i'll do better 🫡", ephemeral=True)


@client.tree.command(name="ratings", description="See your rating history for Corsbot")
async def slash_ratings(interaction: discord.Interaction):
    guild_id = interaction.guild_id if interaction.guild else None
    stats = get_feedback_stats(str(interaction.user.id), guild_id)
    total = stats["good"] + stats["bad"]
    if total == 0:
        await interaction.response.send_message(
            "No ratings yet — use `/rate` after my replies.", ephemeral=True
        )
        return
    pct = int((stats["good"] / total) * 100)
    scope = "this server" if guild_id else "this DM"
    await interaction.response.send_message(
        f"📊 Your ratings {scope}: ✅ {stats['good']} good · ❌ {stats['bad']} bad · {pct}% satisfaction",
        ephemeral=True
    )


@client.tree.command(name="dashboard", description="Show activity dashboard for the bot")
async def slash_dashboard(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Admins only 🔒", ephemeral=True)
        return

    _, cursor = get_db()
    cursor.execute("SELECT content FROM messages WHERE role='user'")
    rows = cursor.fetchall()

    user_counts = Counter()
    for (content,) in rows:
        match = re.match(r"^\[([^\]]+)\]:", content)
        if match:
            user_counts[match.group(1)] += 1

    top_users = user_counts.most_common(5)
    cursor.execute(
        "SELECT key, COUNT(*) FROM memory GROUP BY key ORDER BY COUNT(*) DESC LIMIT 5"
    )
    top_topics = cursor.fetchall()

    cursor.execute(
        "SELECT relation, COUNT(*) FROM relationships GROUP BY relation ORDER BY COUNT(*) DESC LIMIT 5"
    )
    top_relations = cursor.fetchall()

    lines = ["📊 **Dashboard**"]
    if top_users:
        lines.append("**Most active users:**")
        for name, count in top_users:
            lines.append(f"• {name}: {count} messages")
    else:
        lines.append("No user activity recorded yet.")

    if top_topics:
        lines.append("\n**Popular topics:**")
        for key, count in top_topics:
            lines.append(f"• {key}: {count} facts")
    else:
        lines.append("\nNo memory topics recorded yet.")

    if top_relations:
        lines.append("\n**Common relationships:**")
        for relation, count in top_relations:
            lines.append(f"• {relation}: {count} mentions")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@client.tree.command(name="reset", description="Clear your conversation history with Corsbot")
async def slash_reset(interaction: discord.Interaction):
    user_id = interaction.user.id
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    thread_id = get_thread_id(
        user_id,
        interaction.guild_id if interaction.guild else None,
        interaction.channel_id,
        is_dm,
    )
    conn, cursor = get_db()
    cursor.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
    conn.commit()
    await interaction.response.send_message("fresh start 🧹 i remember nothing from this chat", ephemeral=True)


@client.tree.command(name="help", description="Show all Corsbot commands")
async def slash_help(interaction: discord.Interaction):
    lines = [
        "🤖 **Corsbot Commands**",
        "`/memory` — see everything I know about you",
        "`/forget <key>` — delete a specific memory",
        "`/forgetall` — wipe all your memories",
        "`/reset` — clear your conversation history",
        "`/stats` — conversation and memory stats",
        "`/dashboard` — show activity dashboard",
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

    # Cooldown — only applies to messages the bot would actually respond to
    if should_reply:
        now = time.time()
        if now - _user_cooldowns.get(message.author.id, 0) < COOLDOWN_SECONDS:
            return
        _user_cooldowns[message.author.id] = now

    content = message.content.strip()
    ocr_text = ""

    if not should_reply:
        return

    if message.attachments:
        ocr_parts = []
        for attachment in message.attachments:
            extracted = await extract_attachment_text(attachment)
            if extracted:
                ocr_parts.append(extracted)
        if ocr_parts:
            # Sanitize OCR text — image content is untrusted external input
            ocr_text = "\n\n".join(
                sanitize_retrieved_content(part, "ocr") for part in ocr_parts
            ).strip()

    if not content and not ocr_text:
        return

    # Strip bot mention
    content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
    if not content and not ocr_text:
        return

    if ocr_text and content:
        content = f"{ocr_text}\n\n{content}"
    elif ocr_text:
        content = ocr_text

    # Prompt injection guard — regex + semantic
    if is_prompt_injection(content) or is_semantic_jailbreak(content):
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

    loop = asyncio.get_running_loop()

    # Prefix message with sender name so AI never confuses sender with mentioned users
    attributed_content = f"[{message.author.display_name}]: {content}"
    await loop.run_in_executor(executor, store_message, thread_id, "user", attributed_content)

    # Store sender's full identity — display name, username, and server nickname
    guild_nick = message.author.nick if hasattr(message.author, "nick") else None
    await loop.run_in_executor(
        executor, store_user_name,
        message.author.id,
        message.author.display_name,
        message.author.name,
        guild_nick
    )

    # Quick replies
    quick = get_quick_reply(content)
    if quick:
        await message.channel.send(f"<@{message.author.id}> {quick}")
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", quick)
        return

    # ── Agent loop ───────────────────────────────────────────────────────
    uid_str = str(message.author.id)

    # Check for denials before running the agent — user might be correcting a stored fact
    denied_matches = await loop.run_in_executor(executor, check_and_delete_denied_facts, message.author.id, content)
    if denied_matches:
        facts_str = ", ".join(f"`{v}`" for _, v in denied_matches)
        view = DenialConfirmView(message.author.id, denied_matches)
        await message.channel.send(
            f"<@{message.author.id}> you want me to forget {facts_str}? you sure?",
            view=view
        )
        return

    # Background: extract memory + relationships (non-blocking, every N messages)
    if should_extract(uid_str):
        loop.run_in_executor(executor, extract_memory, message.author.id, content)
        loop.run_in_executor(executor, extract_relationships, message.author.id, attributed_content)

    # Background: update reflection periodically
    if should_update_reflection(uid_str):
        async def _bg_reflection():
            recent_history = await loop.run_in_executor(executor, get_history, thread_id, 20)
            recent_user_msgs = [e["content"] for e in recent_history if e["role"] == "user"]
            if recent_user_msgs:
                await loop.run_in_executor(executor, update_reflection, uid_str, recent_user_msgs)
        asyncio.ensure_future(_bg_reflection())

    # Store mentioned users' identity info
    for user in message.mentions:
        if user == client.user:
            continue
        guild_nick_u = user.nick if hasattr(user, "nick") else None
        loop.run_in_executor(executor, store_user_name, user.id, user.display_name, user.name, guild_nick_u)

    # Detect impersonation intent
    impersonate_keywords = ("pretend", "act as", "be ", "impersonate", "roleplay as", "talk like", "speak as")
    is_impersonating = any(kw in content.lower() for kw in impersonate_keywords)

    # Fetch history + feedback context before handing off to agent
    history = await loop.run_in_executor(executor, get_history, thread_id, HISTORY_LIMIT)
    channel_name = message.channel.name if hasattr(message.channel, "name") else "dm"
    feedback_context = await loop.run_in_executor(
        executor, get_feedback_context,
        str(message.author.id),
        message.guild.id if message.guild else None,
    )

    # Build agent context
    from core.agent import AgentContext, AgentLoop
    ctx = AgentContext(
        user_id=message.author.id,
        uid_str=uid_str,
        content=content,
        attributed_content=attributed_content,
        thread_id=thread_id,
        channel_name=channel_name,
        username=message.author.display_name,
        guild_id=message.guild.id if message.guild else None,
        mentioned_users=mentioned_users,
        is_impersonating=is_impersonating,
        history=history,
        feedback_context=feedback_context,
    )

    # Check cache before running the full agent
    cache_key = build_response_cache_key(thread_id, content, "", "", "", feedback_context)
    cached = get_cached_response(cache_key)
    if cached:
        reply = cached
        active_keys = []
    else:
        agent = AgentLoop(executor, loop)
        # Wrap typing() so a Discord rate limit on the typing endpoint
        # doesn't crash the whole handler — the reply still goes out.
        try:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
        except Exception:
            typing_ctx = None

        try:
            ctx, trace = await agent.run(ctx)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in ([^\.]+)", err)
                wait = match.group(1) if match else "a few minutes"
                await message.channel.send(f"<@{message.author.id}> ⏳ rate limited, try again in {wait}.")
            else:
                await message.channel.send(f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}")
                log.error(f"[on_message] agent error: {e}")
            return
        finally:
            if typing_ctx is not None:
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

        if not ctx.reply:
            await message.channel.send(f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}")
            return

        reply = ctx.reply
        active_keys = ctx.active_keys
        set_cached_response(cache_key, reply)

    await loop.run_in_executor(executor, store_message, thread_id, "assistant", reply)
    store_last_reply(str(message.author.id), reply, getattr(ctx, "emotion_state", None) or "default", active_keys)
    reply = resolve_mentions_in_reply(reply, message.guild)

    await send_reply(message.channel, f"<@{message.author.id}> {reply}")

    # Emoji reaction based on detected emotion
    REACTION_MAP = {
        "laugh":     "😂",
        "sad":       "😔",
        "shock":     "😱",
        "angry":     "😤",
        "celebrate": "🎉",
        "happy":     "🥰",
    }
    gif_emotion = getattr(ctx, "gif_emotion", None) if not cached else None
    if gif_emotion and gif_emotion in REACTION_MAP:
        try:
            await message.add_reaction(REACTION_MAP[gif_emotion])
        except Exception:
            pass

    gif_url = getattr(ctx, "gif_url", None)
    if gif_url and not cached:
        await message.channel.send(gif_url)
        await loop.run_in_executor(executor, store_message, thread_id, "assistant", f"[GIF: {gif_emotion}]")

# ---------------- RUN ---------------- #

client.run(DISCORD_TOKEN)
