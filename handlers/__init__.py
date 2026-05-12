"""
Message event handler.
Processes Discord messages and orchestrates the agent pipeline.
"""

import asyncio
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import discord

from core.agent import AgentContext, AgentLoop
from core.ai import FALLBACK_RESPONSES, is_prompt_injection, is_semantic_jailbreak, sanitize_retrieved_content, is_high_risk_intent
from core.db import get_thread_id, get_history, store_message, get_db, store_token_usage
from core.feedback import store_last_reply, get_feedback_context
from core.logger import get_logger
from core.memory import (
    check_and_delete_denied_facts,
    delete_denied_fact,
    extract_memory,
    extract_relationships,
    should_extract,
    should_update_reflection,
    should_purge,
    store_user_name,
    update_reflection,
    purge_stale_memories,
)
from models import DenialConfirmView
from utils import (
    build_response_cache_key,
    extract_attachment_text,
    get_quick_reply,
    resolve_mentions_in_reply,
    send_reply,
    ResponseCache,
)

log = get_logger("corsbot.handlers.messages")

import re as _re

# "mambo" anywhere in the message triggers a GIF
# "2 mambo" or "mambo mambo" sends 2, etc. (capped at 5)
_MAMBO_RE = _re.compile(r'\bmambo\b', _re.I)
_MAMBO_COUNT_RE = _re.compile(r'^(\d+)\s+mambo$', _re.I)

def _extract_gif_request(text: str) -> tuple[str, int] | None:
    """Returns ("mambo", count) if message contains mambo, else None."""
    cleaned = text.strip().rstrip("!?.")

    # "2 mambo" → 2 gifs
    count_match = _MAMBO_COUNT_RE.match(cleaned)
    if count_match:
        return "mambo", min(int(count_match.group(1)), 5)

    # count occurrences of "mambo" in the message
    count = len(_MAMBO_RE.findall(cleaned))
    if count:
        return "mambo", min(count, 5)

    return None

# ────────────────────────────────────────────────────────────────────────────────
# MESSAGE HANDLER
# ────────────────────────────────────────────────────────────────────────────────


class MessageHandler:
    """Handles incoming Discord messages."""
    
    REACTION_MAP = {
        "laugh":     "😂",
        "sad":       "😔",
        "shock":     "😱",
        "angry":     "😤",
        "celebrate": "🎉",
        "happy":     "🥰",
    }
    
    IMPERSONATE_KEYWORDS = ("pretend", "act as", "be ", "impersonate", "roleplay as", "talk like", "speak as")
    
    def __init__(
        self,
        client: discord.Client,
        executor: ThreadPoolExecutor,
        config: dict,
        quick_replies: dict,
        response_cache: ResponseCache = None,
    ):
        self.client = client
        self.executor = executor
        self.config = config
        self.quick_replies = quick_replies
        self.response_cache = response_cache or ResponseCache(ttl_seconds=config.get("RESPONSE_CACHE_TTL", 300))
        self.user_cooldowns: dict[int, float] = {}
    
    async def handle(self, message: discord.Message):
        """Main message handler."""
        if message.author == self.client.user:
            return
        
        is_dm = isinstance(message.channel, discord.DMChannel)
        should_reply = is_dm or (self.client.user in message.mentions)
        
        # Cooldown check
        if should_reply and not await self._check_cooldown(message.author.id):
            return
        
        if not should_reply:
            return
        
        # Extract content and OCR text
        content = message.content.strip()
        ocr_text = ""
        
        if message.attachments:
            ocr_text = await self._extract_attachments(message.attachments)
        
        if not content and not ocr_text:
            return
        
        # Strip bot mention
        content = content.replace(f"<@{self.client.user.id}>", "").replace(f"<@!{self.client.user.id}>", "").strip()
        if not content and not ocr_text:
            return
        
        # Combine OCR and content
        if ocr_text and content:
            content = f"{ocr_text}\n\n{content}"
        elif ocr_text:
            content = ocr_text
        
        # Injection guard — three layers:
        # 1. Regex patterns (fast, zero cost)
        # 2. Semantic similarity (embedding-based, catches indirect framing)
        # 3. Intent classifier (LLM-based, catches creative/scored attacks)
        if is_prompt_injection(content) or is_semantic_jailbreak(content):
            await message.channel.send(f"<@{message.author.id}> nice try 💀")
            return

        blocked, intent = await asyncio.get_running_loop().run_in_executor(
            self.executor, is_high_risk_intent, content
        )
        if blocked:
            await message.channel.send(f"<@{message.author.id}> nice try 💀")
            return
        
        # Resolve mentions
        mentioned_users = await self._resolve_mentions(message, content)
        
        # Get thread ID and store message
        thread_id = get_thread_id(
            message.author.id,
            message.guild.id if message.guild else None,
            message.channel.id,
            is_dm,
        )
        
        loop = asyncio.get_running_loop()
        
        # Store message and user identity
        attributed_content = f"[{message.author.display_name}]: {content}"
        await loop.run_in_executor(self.executor, store_message, thread_id, "user", attributed_content)
        
        guild_nick = message.author.nick if hasattr(message.author, "nick") else None
        await loop.run_in_executor(
            self.executor,
            store_user_name,
            message.author.id,
            message.author.display_name,
            message.author.name,
            guild_nick,
        )
        
        # Quick reply check
        quick = get_quick_reply(content, self.quick_replies)
        if quick:
            await message.channel.send(f"<@{message.author.id}> {quick}")
            await loop.run_in_executor(self.executor, store_message, thread_id, "assistant", quick)
            return

        # Explicit GIF request — "mambo gif", "send gif of cats", "gif pls: dancing"
        gif_request = _extract_gif_request(content)
        if gif_request:
            from core.gif import search_gif
            gif_query, gif_count = gif_request
            sent = 0
            for _ in range(gif_count):
                gif_url = await search_gif(gif_query)
                if gif_url:
                    await message.channel.send(gif_url)
                    sent += 1
            if sent:
                await loop.run_in_executor(self.executor, store_message, thread_id, "assistant", f"[GIF x{sent}: {gif_query}]")
                return
        
        # Denial check
        uid_str = str(message.author.id)
        denied_matches = await loop.run_in_executor(
            self.executor, check_and_delete_denied_facts, message.author.id, content
        )
        if denied_matches:
            await self._handle_denial_confirmation(message, denied_matches)
            return
        
        # Background tasks
        if should_extract(uid_str):
            loop.run_in_executor(self.executor, extract_memory, message.author.id, content)
            loop.run_in_executor(self.executor, extract_relationships, message.author.id, attributed_content)
        
        if should_update_reflection(uid_str):
            asyncio.ensure_future(self._update_reflection_bg(uid_str, thread_id, loop))

        if should_purge(uid_str):
            loop.run_in_executor(self.executor, purge_stale_memories, uid_str)
        
        # Store mentioned users' info
        for user in message.mentions:
            if user == self.client.user:
                continue
            guild_nick_u = user.nick if hasattr(user, "nick") else None
            loop.run_in_executor(
                self.executor, store_user_name, user.id, user.display_name, user.name, guild_nick_u
            )
        
        # Detect impersonation
        is_impersonating = any(kw in content.lower() for kw in self.IMPERSONATE_KEYWORDS)
        
        # Fetch context
        history = await loop.run_in_executor(self.executor, get_history, thread_id, self.config["HISTORY_LIMIT"])
        channel_name = message.channel.name if hasattr(message.channel, "name") else "dm"
        feedback_context = await loop.run_in_executor(
            self.executor, get_feedback_context,
            uid_str,
            message.guild.id if message.guild else None,
        )
        
        # Get user activity
        user_activity, user_status = await self._get_user_presence(message, loop)
        
        # Build agent context
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
            user_activity=user_activity,
            user_status=user_status,
        )
        
        # Check cache
        memory_context = await loop.run_in_executor(
            self.executor,
            lambda: get_db()[1].execute(
                "SELECT GROUP_CONCAT(value, ', ') FROM memory WHERE user_id=? LIMIT 5",
                (uid_str,),
            ).fetchone()[0],
        )
        
        cache_key = build_response_cache_key(thread_id, content, memory_context or "", "", "", feedback_context)
        cached_reply = self.response_cache.get(cache_key)
        
        if cached_reply:
            reply = cached_reply
            active_keys = []
        else:
            # Run agent
            agent = AgentLoop(self.executor, loop)
            reply, active_keys = await self._run_agent(message, agent, ctx, loop)
            if reply:
                self.response_cache.set(cache_key, reply)
        
        if not reply:
            await message.channel.send(f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}")
            return

        # Store and send reply
        await loop.run_in_executor(self.executor, store_message, thread_id, "assistant", reply)
        if not cached_reply:
            store_last_reply(uid_str, reply, getattr(ctx, "emotion_state", None) or "default", active_keys)

        reply = resolve_mentions_in_reply(reply, message.guild)
        await send_reply(message.channel, f"<@{message.author.id}> {reply}")
        
        # Emoji reaction
        gif_emotion = getattr(ctx, "gif_emotion", None) if not cached_reply else None
        if gif_emotion and gif_emotion in self.REACTION_MAP:
            try:
                await message.add_reaction(self.REACTION_MAP[gif_emotion])
            except Exception:
                pass
        
        # Send GIF
        gif_url = getattr(ctx, "gif_url", None)
        if gif_url and not cached_reply:
            await message.channel.send(gif_url)
            await loop.run_in_executor(self.executor, store_message, thread_id, "assistant", f"[GIF: {gif_emotion}]")
    
    async def _check_cooldown(self, user_id: int) -> bool:
        """Check if user is on cooldown."""
        now = time.time()
        last = self.user_cooldowns.get(user_id, 0)
        if now - last < self.config.get("COOLDOWN_SECONDS", 3):
            return False
        self.user_cooldowns[user_id] = now
        return True
    
    async def _extract_attachments(self, attachments) -> str:
        """Extract text from attachments."""
        ocr_parts = []
        for attachment in attachments:
            extracted = await extract_attachment_text(
                attachment,
                self.executor,
                self.config.get("MAX_IMAGE_BYTES", 8 * 1024 * 1024),
            )
            if extracted:
                ocr_parts.append(extracted)
        
        if ocr_parts:
            return "\n\n".join(
                sanitize_retrieved_content(part, "ocr") for part in ocr_parts
            ).strip()
        return ""
    
    async def _resolve_mentions(self, message: discord.Message, content: str) -> dict:
        """Resolve and replace user mentions."""
        mentioned_users = {}
        for user in message.mentions:
            if user == self.client.user:
                continue
            display = user.display_name
            content = content.replace(f"<@{user.id}>", f"@{display}").replace(f"<@!{user.id}>", f"@{display}")
            mentioned_users[user.id] = display
        return mentioned_users
    
    async def _handle_denial_confirmation(self, message: discord.Message, denied_matches: list):
        """Show denial confirmation view."""
        facts_str = ", ".join(f"`{v}`" for _, v in denied_matches)
        view = DenialConfirmView(message.author.id, denied_matches, delete_denied_fact)
        await message.channel.send(
            f"<@{message.author.id}> you want me to forget {facts_str}? you sure?",
            view=view,
        )
    
    async def _update_reflection_bg(self, uid_str: str, thread_id: str, loop: asyncio.AbstractEventLoop):
        """Background task to update reflection."""
        recent_history = await loop.run_in_executor(self.executor, get_history, thread_id, 20)
        recent_user_msgs = [e["content"] for e in recent_history if e["role"] == "user"]
        if recent_user_msgs:
            await loop.run_in_executor(self.executor, update_reflection, uid_str, recent_user_msgs)
    
    async def _get_user_presence(self, message: discord.Message, loop: asyncio.AbstractEventLoop) -> tuple[str, str]:
        """Extract user's current activity and status."""
        user_activity = ""
        user_status = ""
        
        if not message.guild:
            return user_activity, user_status
        
        user_member = message.guild.get_member(message.author.id)
        if not user_member:
            return user_activity, user_status
        
        if user_member.activity:
            activity = user_member.activity
            if hasattr(activity, 'details') and activity.details:
                if hasattr(activity, 'state') and activity.state:
                    user_activity = f"{activity.name}: {activity.state} - {activity.details}"
                else:
                    user_activity = f"{activity.name}: {activity.details}"
            else:
                user_activity = activity.name
        
        user_status = str(user_member.status)
        return user_activity, user_status
    
    async def _run_agent(self, message: discord.Message, agent: AgentLoop, ctx: AgentContext, loop: asyncio.AbstractEventLoop) -> tuple[str, list]:
        """Run the agent and handle errors."""
        typing_ctx = None
        try:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
        except Exception:
            pass
        
        try:
            ctx, trace = await agent.run(ctx)
            return ctx.reply, ctx.active_keys
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in ([^\.]+)", err)
                wait = match.group(1) if match else "a few minutes"
                await message.channel.send(f"<@{message.author.id}> ⏳ rate limited, try again in {wait}.")
            else:
                await message.channel.send(f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}")
                log.error("agent_failed",
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else None,
                    error=str(e),
                )
            return "", []
        finally:
            if typing_ctx is not None:
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
