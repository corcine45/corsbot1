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
from core.ai import (
    FALLBACK_RESPONSES,
    is_high_risk_intent,
    is_prompt_injection,
    is_semantic_jailbreak,
    sanitize_retrieved_content,
)
from core.db import (
    get_conversation_thread_id,
    get_db,
    get_history,
    get_recent_speakers,
    get_thread_id,
    store_message,
    store_token_usage,
)
from core.feedback import get_feedback_context, store_last_reply
from core.instructions import parse_deferred_instruction, store_instruction
from core.logger import get_logger
from core.memory import (
    check_and_delete_denied_facts,
    delete_denied_fact,
    extract_memory,
    extract_relationships,
    purge_stale_memories,
    should_extract,
    should_purge,
    should_update_reflection,
    store_user_name,
    update_reflection,
)
from models import DenialConfirmView
from utils import (
    ResponseCache,
    build_response_cache_key,
    extract_attachment_text,
    extract_video_description,
    get_quick_reply,
    resolve_mentions_in_reply,
    resolve_message_channel,
    send_reply,
)

log = get_logger("corsbot.handlers.messages")

import re as _re

# "mambo" anywhere in the message triggers a GIF
# "2 mambo" or "mambo mambo" sends 2, etc. (capped at 5)
_MAMBO_RE = _re.compile(r"\bmamboo*\b", _re.I)
_MAMBO_COUNT_RE = _re.compile(r"^(\d+)\s+mamboo*$", _re.I)
_NEGATIVE_MAMBO_RE = _re.compile(
    r"\b(?:no|not|never|none|without|hardly|barely|scarcely|ain't|aint|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent|don't|dont|doesn't|doesnt|didn't|didnt|can't|cant|couldn't|couldnt|won't|wont|shouldn't|shouldnt|wouldn't|wouldnt)\b(?:[\s\W]+\w+){0,3}[\s\W]+\bmamboo*\b"
    r"|\bmamboo*\b(?:[\s\W]+\w+){0,3}[\s\W]+\b(?:no|not|never|none|without|hardly|barely|scarcely|ain't|aint|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent|don't|dont|doesn't|doesnt|didn't|didnt|can't|cant|couldn't|couldnt|won't|wont|shouldn't|shouldnt|wouldn't|wouldnt)\b",
    _re.I,
)


def _normalize_member_lookup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _resolve_deferred_target(
    guild: discord.Guild, raw_target: str
) -> tuple[discord.Member | None, bool]:
    target = raw_target.strip()
    if not target:
        return None, False

    mention_match = re.fullmatch(r"<@!?(\d+)>", target)
    if mention_match:
        member = guild.get_member(int(mention_match.group(1)))
        return member, member is not None

    normalized_target = _normalize_member_lookup(target.lstrip("@"))
    exact_matches = []
    partial_matches = []

    for member in guild.members:
        if member.bot:
            continue
        candidate_names = {
            _normalize_member_lookup(member.display_name),
            _normalize_member_lookup(member.name),
            _normalize_member_lookup(str(member)),
        }
        if normalized_target in candidate_names:
            exact_matches.append(member)
        elif any(normalized_target in candidate for candidate in candidate_names):
            partial_matches.append(member)

    if len(exact_matches) == 1:
        return exact_matches[0], True
    if len(exact_matches) > 1:
        return None, True
    if len(partial_matches) == 1:
        return partial_matches[0], True
    if len(partial_matches) > 1:
        return None, True
    return None, False


def _extract_gif_request(text: str) -> tuple[str, int] | None:
    """Returns ("mambo", count) if message contains mambo, else None."""
    cleaned = text.strip().rstrip("!?.")

    # If the user explicitly negates mambo, don't treat it as a GIF request.
    if _NEGATIVE_MAMBO_RE.search(cleaned):
        return None

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
        "laugh": "😂",
        "sad": "😔",
        "shock": "😱",
        "angry": "😤",
        "celebrate": "🎉",
        "happy": "🥰",
    }

    IMPERSONATE_KEYWORDS = (
        "pretend",
        "act as",
        "be ",
        "impersonate",
        "roleplay as",
        "talk like",
        "speak as",
    )

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
        self.response_cache = response_cache or ResponseCache(
            ttl_seconds=config.get("RESPONSE_CACHE_TTL", 300)
        )
        self.user_cooldowns: dict[int, float] = {}

    async def _store_message_in_threads(
        self, loop, thread_ids: list[str], role: str, content: str
    ):
        for thread_id in dict.fromkeys(thread_ids):
            await loop.run_in_executor(
                self.executor, store_message, thread_id, role, content
            )

    def _is_recent_speaker_question(self, content: str) -> bool:
        text = content.lower()
        if not any(
            phrase in text
            for phrase in (
                "who were you talking to",
                "who was it talking to",
                "who did you talk to",
                "who were u talking to",
            )
        ):
            return False
        return any(
            phrase in text
            for phrase in (
                "last minute",
                "last min",
                "past minute",
                "1 minute",
                "recently",
                "just now",
                "a minute ago",
            )
        )

    async def handle(self, message: discord.Message):
        """Main message handler."""
        if message.author == self.client.user:
            return

        ch = resolve_message_channel(message.channel)
        is_dm = ch["is_dm"]
        discord_thread_id = ch["discord_thread_id"]
        channel_id = ch["channel_id"]
        guild_id = ch["guild_id"]
        should_reply = (self.client.user in message.mentions) and not is_dm

        # Extract content and OCR text early so mambo GIFs can auto-trigger in public chat.
        content = message.content.strip()
        ocr_text = ""

        # Transcribe voice/audio attachments
        for attachment in message.attachments:
            from core.transcribe import is_audio_attachment, transcribe_discord_attachment

            if is_audio_attachment(attachment.filename, attachment.content_type):
                transcript = await transcribe_discord_attachment(attachment)
                if transcript:
                    content = (
                        f"{content}\n\n[Voice: {transcript}]".strip()
                        if content
                        else f"[Voice: {transcript}]"
                    )

        if message.attachments:
            # Get recent history for better image context detection
            thread_id_for_context = get_conversation_thread_id(
                message.author.id,
                guild_id,
                channel_id,
                is_dm,
                discord_thread_id=discord_thread_id,
            )
            loop = asyncio.get_running_loop()
            recent_history = await loop.run_in_executor(
                self.executor, get_history, thread_id_for_context, 10
            )
            conversation_context = (
                "\n".join(entry.get("content", "") for entry in recent_history[-5:])
                if recent_history
                else ""
            )

            ocr_text = await self._extract_attachments(
                message.attachments,
                message_content=content,
                conversation_context=conversation_context,
            )

        if not content and not ocr_text:
            return

        # Strip bot mention
        content = (
            content.replace(f"<@{self.client.user.id}>", "")
            .replace(f"<@!{self.client.user.id}>", "")
            .strip()
        )
        if not content and not ocr_text:
            return

        # Combine OCR and content
        if ocr_text and content:
            content = f"{ocr_text}\n\n{content}"
        elif ocr_text:
            content = ocr_text

        # Explicit GIF request — mambo anywhere in message
        gif_request = _extract_gif_request(content)
        if gif_request:
            from core.gif import search_gif

            gif_query, gif_count = gif_request
            sent = 0
            loop = asyncio.get_running_loop()
            personal_thread_id = get_thread_id(
                message.author.id,
                guild_id,
                channel_id,
                is_dm,
                discord_thread_id=discord_thread_id,
            )
            thread_id = get_conversation_thread_id(
                message.author.id,
                guild_id,
                channel_id,
                is_dm,
                discord_thread_id=discord_thread_id,
            )
            for _ in range(gif_count):
                gif_url = await search_gif(gif_query)
                if gif_url:
                    await message.channel.send(gif_url)
                    sent += 1
                else:
                    log.warning("gif_not_found", query=gif_query)
            if sent:
                await self._store_message_in_threads(
                    loop,
                    [thread_id, personal_thread_id],
                    "assistant",
                    f"[GIF x{sent}: {gif_query}]",
                )
                return
            # GIF search failed — fall through to normal reply

        loop = asyncio.get_running_loop()
        personal_thread_id = get_thread_id(
            message.author.id,
            guild_id,
            channel_id,
            is_dm,
            discord_thread_id=discord_thread_id,
        )
        thread_id = get_conversation_thread_id(
            message.author.id,
            guild_id,
            channel_id,
            is_dm,
            discord_thread_id=discord_thread_id,
        )

        # Thread-aware: keep replying in active bot threads without re-mentioning
        if ch["is_thread"] and not should_reply:
            history = await loop.run_in_executor(
                self.executor, get_history, thread_id, 1
            )
            if history:
                should_reply = True

        if not should_reply:
            return

        from core.guild_settings import is_channel_opted_out, memory_user_key

        if is_channel_opted_out(guild_id, channel_id):
            return

        uid_str = str(message.author.id)
        memory_uid = memory_user_key(message.author.id, guild_id)

        if not await self._check_cooldown(message.author.id):
            return

        from core.proactivity import note_user_interaction

        note_user_interaction(message.author.id, message.channel.id, guild_id)

        # Injection guard: regex, semantic similarity, and intent classifier.
        try:
            injection_blocked = is_prompt_injection(content) or is_semantic_jailbreak(content)
        except Exception as exc:
            log.warning("injection_guard_failed", error=str(exc))
            injection_blocked = False

        if injection_blocked:
            await message.channel.send(f"<@{message.author.id}> nice try 💀")
            return

        blocked, intent = await asyncio.get_running_loop().run_in_executor(
            self.executor, is_high_risk_intent, content
        )
        if blocked:
            await message.channel.send(f"<@{message.author.id}> nice try 💀")
            return

        # Resolve mentions
        content, mentioned_users = await self._resolve_mentions(message, content)

        # Quick reply check
        quick = get_quick_reply(content, self.quick_replies)
        if quick:
            attributed_content = f"[{message.author.display_name}]: {content}"
            await self._store_message_in_threads(
                loop, [thread_id, personal_thread_id], "user", attributed_content
            )
            await loop.run_in_executor(
                self.executor,
                store_user_name,
                message.author.id,
                message.author.display_name,
                message.author.name,
                message.author.nick if hasattr(message.author, "nick") else None,
            )
            await message.channel.send(f"<@{message.author.id}> {quick}")
            await self._store_message_in_threads(
                loop, [thread_id, personal_thread_id], "assistant", quick
            )
            return

        # Store message and user identity
        attributed_content = f"[{message.author.display_name}]: {content}"
        await self._store_message_in_threads(
            loop, [thread_id, personal_thread_id], "user", attributed_content
        )

        guild_nick = message.author.nick if hasattr(message.author, "nick") else None
        await loop.run_in_executor(
            self.executor,
            store_user_name,
            message.author.id,
            message.author.display_name,
            message.author.name,
            guild_nick,
        )

        # Scheduled reminder — "remind me in 30 minutes to ..."
        from core.reminders import parse_reminder, store_reminder

        parsed_reminder = parse_reminder(content)
        if parsed_reminder:
            fire_at, reminder_msg = parsed_reminder
            store_reminder(
                message.author.id,
                guild_id,
                message.channel.id,
                reminder_msg,
                fire_at,
            )
            mins = max(1, round((fire_at - time.time()) / 60))
            await message.channel.send(
                f"<@{message.author.id}> got it — I'll ping you in ~{mins} min: **{reminder_msg[:120]}**"
            )
            return

        # Deferred instruction detection — "when X comes online, do Y"
        if message.guild:
            parsed = parse_deferred_instruction(content)
            if parsed:
                trigger_target, action = parsed
                resolved_member, had_matches = _resolve_deferred_target(
                    message.guild, trigger_target
                )
                store_instruction(
                    message.author.id,
                    message.guild.id,
                    message.channel.id,
                    "online",
                    trigger_target,
                    action,
                    trigger_target_id=resolved_member.id if resolved_member else None,
                )
                if resolved_member:
                    target_label = resolved_member.display_name
                    await message.channel.send(
                        f"<@{message.author.id}> got it — I'll say something when **{target_label}** comes online 👀"
                    )
                elif had_matches:
                    await message.channel.send(
                        f"<@{message.author.id}> got it — I saved that, but **{trigger_target}** matches multiple people, so I kept it name-based. If you want one specific person, @mention them."
                    )
                else:
                    await message.channel.send(
                        f"<@{message.author.id}> got it — I couldn't uniquely identify **{trigger_target}**, so I saved it name-based. If you want one specific person, @mention them."
                    )
                return

        if self._is_recent_speaker_question(content):
            speakers = await loop.run_in_executor(
                self.executor,
                get_recent_speakers,
                thread_id,
                60,
                [message.author.display_name, message.author.name],
            )
            if speakers:
                names = ", ".join(speakers[:4])
                reply = f"In the last minute, I was talking to {names}."
            else:
                reply = "I don't see anyone else talking to me in the last minute."
            await message.channel.send(f"<@{message.author.id}> {reply}")
            await self._store_message_in_threads(
                loop, [thread_id, personal_thread_id], "assistant", reply
            )
            return

        # Denial check
        denied_matches = await loop.run_in_executor(
            self.executor,
            check_and_delete_denied_facts,
            message.author.id,
            content,
            thread_id,
        )
        if denied_matches:
            await self._handle_denial_confirmation(message, denied_matches)
            return

        # Background tasks
        if should_extract(uid_str):
            loop.run_in_executor(
                self.executor, extract_memory, memory_uid, content
            )
            loop.run_in_executor(
                self.executor,
                extract_relationships,
                memory_uid,
                attributed_content,
            )
            from core.memory import extract_relationship_categories_from_message

            loop.run_in_executor(
                self.executor,
                extract_relationship_categories_from_message,
                memory_uid,
                content,
            )

        if should_update_reflection(uid_str):
            asyncio.ensure_future(
                self._update_reflection_bg(uid_str, personal_thread_id, loop)
            )

        if should_purge(uid_str):
            loop.run_in_executor(self.executor, purge_stale_memories, uid_str)

        # Store mentioned users' info
        for user in message.mentions:
            if user == self.client.user:
                continue
            guild_nick_u = user.nick if hasattr(user, "nick") else None
            loop.run_in_executor(
                self.executor,
                store_user_name,
                user.id,
                user.display_name,
                user.name,
                guild_nick_u,
            )

        # Detect impersonation
        is_impersonating = any(
            kw in content.lower() for kw in self.IMPERSONATE_KEYWORDS
        )

        # Fetch context in parallel
        channel_name = (
            message.channel.name if hasattr(message.channel, "name") else "dm"
        )

        def _fetch_memory_context():
            return (
                get_db()[1]
                .execute(
                    "SELECT GROUP_CONCAT(value, ', ') FROM memory WHERE user_id=? LIMIT 5",
                    (memory_uid,),
                )
                .fetchone()[0]
            )

        history, feedback_context, (user_activity, user_status), memory_context = (
            await asyncio.gather(
                loop.run_in_executor(
                    self.executor, get_history, thread_id, self.config["HISTORY_LIMIT"]
                ),
                loop.run_in_executor(
                    self.executor,
                    get_feedback_context,
                    uid_str,
                    message.guild.id if message.guild else None,
                ),
                self._get_user_presence(message, loop),
                loop.run_in_executor(self.executor, _fetch_memory_context),
            )
        )

        # Build member list for small servers (skip bots)
        server_members = ""
        if message.guild and len(message.guild.members) <= 30:
            names = [m.display_name for m in message.guild.members if not m.bot]
            if names:
                server_members = ", ".join(names)

        # Build agent context
        ctx = AgentContext(
            user_id=message.author.id,
            uid_str=uid_str,
            memory_user_id=memory_uid,
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
            server_members=server_members,
        )

        # Check cache
        history_context = "\n".join(
            f"{entry.get('role', '')}:{entry.get('content', '')[:180]}"
            for entry in history[-6:]
        )
        cache_key = build_response_cache_key(
            thread_id,
            content,
            memory_context or "",
            feedback_context=feedback_context,
            history_context=history_context,
        )
        cached_reply = self.response_cache.get(cache_key)
        agent = None

        if cached_reply:
            reply = cached_reply
            active_keys = []
        else:
            agent = AgentLoop(self.executor, loop)
            reply, active_keys, ctx = await self._run_agent(
                message, agent, ctx, loop
            )
            if reply:
                self.response_cache.set(cache_key, reply)

        if not reply:
            await message.channel.send(
                f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}"
            )
            return

        # Store and send reply
        await self._store_message_in_threads(
            loop, [thread_id, personal_thread_id], "assistant", reply
        )
        if not cached_reply:
            store_last_reply(
                uid_str,
                reply,
                getattr(ctx, "emotion_state", None) or "default",
                active_keys,
            )

        reply = resolve_mentions_in_reply(reply, message.guild, message.author.id)
        await send_reply(message.channel, f"<@{message.author.id}> {reply}")

        if agent and not cached_reply and reply:
            await agent.step_post(ctx, None)

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
            await self._store_message_in_threads(
                loop,
                [thread_id, personal_thread_id],
                "assistant",
                f"[GIF: {gif_emotion}]",
            )

    async def _check_cooldown(self, user_id: int) -> bool:
        """Check if user is on cooldown."""
        now = time.time()
        last = self.user_cooldowns.get(user_id, 0)
        if now - last < self.config.get("COOLDOWN_SECONDS", 3):
            return False
        self.user_cooldowns[user_id] = now
        return True

    async def _extract_attachments(
        self, attachments, message_content: str = "", conversation_context: str = ""
    ) -> str:
        """
        Extract text/descriptions from attachments with context-aware analysis.
        Handles both images and videos.

        Args:
            attachments: List of Discord attachment objects
            message_content: The text message that accompanied the attachments
            conversation_context: Recent conversation history for better analysis
        """
        parts = []
        for attachment in attachments:
            # Check if it's a video
            if attachment.content_type and attachment.content_type.startswith("video/"):
                extracted = await extract_video_description(
                    attachment,
                    self.executor,
                    message_content=message_content,
                    conversation_context=conversation_context,
                )
            else:
                # Handle images and other attachments
                extracted = await extract_attachment_text(
                    attachment,
                    self.executor,
                    self.config.get("MAX_IMAGE_BYTES", 8 * 1024 * 1024),
                    message_content=message_content,
                    conversation_context=conversation_context,
                )
            if extracted:
                parts.append(extracted)

        if parts:
            return "\n\n".join(
                sanitize_retrieved_content(part, "attachment") for part in parts
            ).strip()
        return ""

    async def _resolve_mentions(
        self, message: discord.Message, content: str
    ) -> tuple[str, dict[int, str]]:
        """Resolve and replace user mentions."""
        mentioned_users = {}
        for user in message.mentions:
            if user == self.client.user:
                continue
            display = user.display_name
            content = content.replace(f"<@{user.id}>", f"@{display}").replace(
                f"<@!{user.id}>", f"@{display}"
            )
            mentioned_users[user.id] = display
        return content, mentioned_users

    async def _handle_denial_confirmation(
        self, message: discord.Message, denied_matches: list
    ):
        """Show denial confirmation view."""
        facts_str = ", ".join(f"`{v}`" for _, v in denied_matches)
        view = DenialConfirmView(message.author.id, denied_matches, delete_denied_fact)
        await message.channel.send(
            f"<@{message.author.id}> you want me to forget {facts_str}? you sure?",
            view=view,
        )

    async def _update_reflection_bg(
        self, uid_str: str, thread_id: str, loop: asyncio.AbstractEventLoop
    ):
        """Background task to update reflection."""
        recent_history = await loop.run_in_executor(
            self.executor, get_history, thread_id, 20
        )
        recent_user_msgs = [e["content"] for e in recent_history if e["role"] == "user"]
        if recent_user_msgs:
            await loop.run_in_executor(
                self.executor, update_reflection, uid_str, recent_user_msgs
            )

    async def _get_user_presence(
        self, message: discord.Message, loop: asyncio.AbstractEventLoop
    ) -> tuple[str, str]:
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
            log.debug(
                f"Activity type: {type(activity).__name__}, name: {getattr(activity, 'name', '?')}"
            )
            if isinstance(activity, discord.Spotify):
                title = getattr(activity, "title", None)
                artist = getattr(activity, "artist", None)
                log.debug(f"Spotify detected - title: {title}, artist: {artist}")
                if title and artist:
                    user_activity = f"Spotify: {title} by {artist}"
                else:
                    user_activity = "Spotify"
            elif hasattr(activity, "details") and activity.details:
                if hasattr(activity, "state") and activity.state:
                    user_activity = (
                        f"{activity.name}: {activity.state} - {activity.details}"
                    )
                else:
                    user_activity = f"{activity.name}: {activity.details}"
            else:
                user_activity = activity.name

        user_status = str(user_member.status)
        return user_activity, user_status

    async def _run_agent(
        self,
        message: discord.Message,
        agent: AgentLoop,
        ctx: AgentContext,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[str, list, AgentContext]:
        """Run the agent and handle errors."""
        typing_ctx = None
        try:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
        except Exception:
            pass

        try:
            ctx, trace = await agent.run(ctx)
            return ctx.reply, ctx.active_keys, ctx
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in ([^\.]+)", err)
                wait = match.group(1) if match else "a few minutes"
                await message.channel.send(
                    f"<@{message.author.id}> ⏳ rate limited, try again in {wait}."
                )
            else:
                await message.channel.send(
                    f"<@{message.author.id}> {random.choice(FALLBACK_RESPONSES)}"
                )
                log.error(
                    "agent_failed",
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else None,
                    error=str(e),
                )
            return "", [], ctx
        finally:
            if typing_ctx is not None:
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
