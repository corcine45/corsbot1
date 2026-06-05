"""
Corsbot Main Entry Point
Orchestrates Discord client, handlers, and core systems.
"""

import asyncio
import logging
import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import discord
from discord import app_commands

import config
from handlers import MessageHandler
from handlers.commands import CommandsHandler
from utils import ResponseCache
from core.memory import start_faiss_rebuild_background
from core.presence import describe_activity, record_presence_pattern

log = logging.getLogger("corsbot")

# ── Load libopus explicitly for Railway/Linux ────────────────────────────────
def _load_opus():
    if discord.opus.is_loaded():
        return
    # Common paths on Nix/Railway
    candidates = [
        "libopus.so.0",
        "libopus.so",
        "/nix/store/libopus/lib/libopus.so.0",  # nixpacks path pattern
        ctypes.util.find_library("opus"),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            discord.opus.load_opus(path)
            log.info(f"Loaded opus from: {path}")
            return
        except Exception:
            continue
    log.warning("Could not load libopus — voice will not work")

_load_opus()

# ────────────────────────────────────────────────────────────────────────────────
# CLIENT SETUP
# ────────────────────────────────────────────────────────────────────────────────


class CorsBot(discord.Client):
    """Discord client for Corsbot."""
    
    def __init__(self, executor: ThreadPoolExecutor, response_cache: ResponseCache):
        intents = discord.Intents.default()
        intents.message_content = config.INTENTS["message_content"]
        intents.members = config.INTENTS["members"]
        intents.presences = config.INTENTS["presences"]
        intents.voice_states = config.INTENTS["voice_states"]
        
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.executor = executor
        self.response_cache = response_cache
        self.message_handler = None
        self.commands_handler = None
    
    async def setup_hook(self):
        """Called before the bot connects."""
        try:
            await self.tree.sync()
            log.info("✅ Slash commands synced.")
        except Exception as e:
            import traceback
            print(f"FATAL: setup_hook failed: {e}")
            traceback.print_exc()
            raise
    
    async def on_ready(self):
        """Called when the bot has connected."""
        try:
            log.info(f"✅ Logged in as {self.user}")
            start_faiss_rebuild_background()
        except Exception as e:
            import traceback
            print(f"FATAL: on_ready failed: {e}")
            traceback.print_exc()
        start_faiss_rebuild_background()
    
    async def on_presence_update(self, before: discord.Member, after: discord.Member):
        """Track presence changes (activity/status)."""
        if before.activity != after.activity:
            activity_str = describe_activity(after.activity)
            if activity_str:
                is_social = bool(
                    after.voice
                    and after.voice.channel
                    and len([m for m in after.voice.channel.members if not m.bot]) > 1
                )
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    self.executor,
                    partial(record_presence_pattern, after.id, "activity", activity_str, is_social=is_social),
                )
                log.debug(f"{after.name} presence pattern updated: {activity_str}")
    
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        """Track status changes."""
        if before.status != after.status:
            status = str(after.status)
            if status != "offline":
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    self.executor,
                    record_presence_pattern,
                    after.id,
                    "status",
                    status,
                )
            log.debug(f"{after.name}'s status changed: {before.status} → {after.status}")
    
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """Track voice chat activity."""
        if before.channel == after.channel or after.channel is None:
            return
        
        is_social = len([m for m in after.channel.members if not m.bot]) > 1
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            self.executor,
            partial(record_presence_pattern, member.id, "voice", "voice chat", is_social=is_social),
        )
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        if self.message_handler:
            await self.message_handler.handle(message)


# ────────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ────────────────────────────────────────────────────────────────────────────────


def create_bot() -> CorsBot:
    """Create and configure the bot instance."""
    try:
        executor = ThreadPoolExecutor(max_workers=4)
        response_cache = ResponseCache(ttl_seconds=config.RESPONSE_CACHE_TTL)
        
        bot = CorsBot(executor, response_cache)
        
        bot.message_handler = MessageHandler(
            client=bot,
            executor=executor,
            config={
                "HISTORY_LIMIT": config.HISTORY_LIMIT,
                "COOLDOWN_SECONDS": config.COOLDOWN_SECONDS,
                "RESPONSE_CACHE_TTL": config.RESPONSE_CACHE_TTL,
                "MAX_IMAGE_BYTES": config.MAX_IMAGE_BYTES,
            },
            quick_replies=config.QUICK_REPLIES,
            response_cache=response_cache,
        )
        
        bot.commands_handler = CommandsHandler(bot, executor)
        bot.commands_handler.register_all()
        
        return bot
    except Exception as e:
        import traceback
        print(f"FATAL: create_bot() failed: {e}")
        traceback.print_exc()
        raise


# ────────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        bot = create_bot()
        bot.run(config.DISCORD_TOKEN)
    except Exception as e:
        import traceback
        print(f"FATAL STARTUP ERROR: {e}")
        traceback.print_exc()
