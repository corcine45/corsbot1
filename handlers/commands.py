"""
Slash commands handler.
Implements all /memory, /rate, /stats, etc. commands.
"""

import logging
import time
import re
from collections import Counter
from typing import Callable

import discord
from discord import app_commands

from core.db import get_db, get_thread_id, get_token_stats
from core.feedback import (
    get_feedback_stats,
    get_last_reply,
    store_feedback,
    apply_good_rating,
    apply_bad_rating,
)
from core.memory import delete_denied_fact
from models import DenialConfirmView
from utils import send_interaction

log = logging.getLogger("corsbot.handlers.commands")


# ────────────────────────────────────────────────────────────────────────────────
# COMMANDS HANDLER
# ────────────────────────────────────────────────────────────────────────────────


class CommandsHandler:
    """Registers and handles slash commands."""
    
    def __init__(self, client: discord.Client, executor):
        self.client = client
        self.executor = executor
        self.tree = client.tree
    
    def register_all(self):
        """Register all slash commands."""
        self.tree.command(name="memory", description="See everything Corsbot knows about you")(self.memory)
        self.tree.command(name="forget", description="Delete a specific memory by key")(self.forget)
        self.tree.command(name="forgetall", description="Wipe everything Corsbot knows about you")(self.forgetall)
        self.tree.command(name="stats", description="Show conversation and memory stats")(self.stats)
        self.tree.command(name="relationships", description="See people Corsbot knows about in your life")(self.relationships)
        self.tree.command(name="rate", description="Rate Corsbot's last reply")(self.rate)
        self.tree.command(name="ratings", description="See your rating history for Corsbot")(self.ratings)
        self.tree.command(name="tokens", description="Show your API token usage")(self.tokens)
        self.tree.command(name="dashboard", description="Show activity dashboard for the bot")(self.dashboard)
        self.tree.command(name="reset", description="Clear your conversation history with Corsbot")(self.reset)
        self.tree.command(name="help", description="Show all Corsbot commands")(self.help_command)
    
    async def memory(self, interaction: discord.Interaction):
        """Show all stored memories for the user."""
        user_id = interaction.user.id
        _, cursor = get_db()
        cursor.execute(
            "SELECT key, value, memory_type, reinforcement, updated_at FROM memory WHERE user_id=? ORDER BY memory_type, key",
            (str(user_id),),
        )
        rows = cursor.fetchall()
        
        if not rows:
            await interaction.response.send_message(
                "I don't know anything about you yet.",
                ephemeral=True,
            )
            return
        
        lines = ["📋 **What I know about you:**"]
        for key, value, mtype, reinforcement, updated_at in rows:
            age_days = (time.time() - updated_at) / 86400
            icon = {"identity": "🔒", "preference": "⭐", "temporary": "⏳"}.get(mtype or "preference", "•")
            lines.append(f"{icon} `{key}` = {value}  _(seen {reinforcement}x, {age_days:.0f}d ago)_")
        
        await send_interaction(interaction, "\n".join(lines))
    
    @app_commands.describe(key="The memory key to delete (use /memory to see keys)")
    async def forget(self, interaction: discord.Interaction, key: str):
        """Delete a specific memory."""
        user_id = interaction.user.id
        conn, cursor = get_db()
        cursor.execute("DELETE FROM memory WHERE user_id=? AND key=?", (str(user_id), key.lower()))
        conn.commit()
        
        if cursor.rowcount:
            await interaction.response.send_message(f"Forgot `{key}`. ✅", ephemeral=True)
        else:
            await interaction.response.send_message(f"Nothing stored under `{key}`.", ephemeral=True)
    
    async def forgetall(self, interaction: discord.Interaction):
        """Wipe all memories and relationships."""
        user_id = interaction.user.id
        conn, cursor = get_db()
        cursor.execute("DELETE FROM memory WHERE user_id=?", (str(user_id),))
        cursor.execute("DELETE FROM relationships WHERE user_id=?", (str(user_id),))
        conn.commit()
        
        await interaction.response.send_message(
            "Cleared all your memories and relationships. Fresh start. 🧹",
            ephemeral=True,
        )
    
    async def stats(self, interaction: discord.Interaction):
        """Show conversation and memory stats."""
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
        
        cursor.execute(
            "SELECT COUNT(*), memory_type FROM memory WHERE user_id=? GROUP BY memory_type",
            (str(user_id),),
        )
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
    
    async def relationships(self, interaction: discord.Interaction):
        """Show relationships in the user's life."""
        user_id = interaction.user.id
        _, cursor = get_db()
        cursor.execute(
            "SELECT related_name, relation, context, strength, updated_at FROM relationships WHERE user_id=? ORDER BY strength DESC",
            (str(user_id),),
        )
        rows = cursor.fetchall()
        
        if not rows:
            await interaction.response.send_message(
                "I don't know anyone in your life yet — tell me about your friends!",
                ephemeral=True,
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
    
    @app_commands.describe(rating="Was the reply good or bad?")
    @app_commands.choices(
        rating=[
            app_commands.Choice(name="👍 Good", value="good"),
            app_commands.Choice(name="👎 Bad", value="bad"),
        ]
    )
    async def rate(self, interaction: discord.Interaction, rating: str):
        """Rate the bot's last reply."""
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
                "No recent reply to rate — ratings expire after 5 minutes.",
                ephemeral=True,
            )
            return
        
        store_feedback(
            str(user_id),
            thread_id,
            entry["reply"],
            rating,
            entry["mood"],
            interaction.guild_id if interaction.guild else None,
        )
        
        if rating == "good":
            apply_good_rating(str(user_id), entry["mood"], entry["memory_keys"])
            await interaction.response.send_message("glad you liked it 🙏", ephemeral=True)
        else:
            apply_bad_rating(str(user_id), entry["mood"])
            await interaction.response.send_message("noted, i'll do better 🫡", ephemeral=True)
    
    async def ratings(self, interaction: discord.Interaction):
        """Show rating history."""
        guild_id = interaction.guild_id if interaction.guild else None
        stats = get_feedback_stats(str(interaction.user.id), guild_id)
        total = stats["good"] + stats["bad"]
        
        if total == 0:
            await interaction.response.send_message(
                "No ratings yet — use `/rate` after my replies.",
                ephemeral=True,
            )
            return
        
        pct = int((stats["good"] / total) * 100)
        scope = "this server" if guild_id else "this DM"
        await interaction.response.send_message(
            f"📊 Your ratings {scope}: ✅ {stats['good']} good · ❌ {stats['bad']} bad · {pct}% satisfaction",
            ephemeral=True,
        )
    
    async def tokens(self, interaction: discord.Interaction):
        """Show API token usage."""
        stats = get_token_stats(interaction.user.id)
        lines = [
            "📊 **Token Usage**",
            f"Total tokens used: **{stats['total']:,}**",
            f"Tokens used today: **{stats['today']:,}**",
            f"Total API requests: **{stats['requests']}**",
        ]
        await send_interaction(interaction, "\n".join(lines))
    
    async def dashboard(self, interaction: discord.Interaction):
        """Show activity dashboard (admin only)."""
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("Admins only 🔒", ephemeral=True)
            return
        
        _, cursor = get_db()
        cursor.execute("SELECT content FROM messages WHERE role='user'")
        rows = cursor.fetchall()
        
        # Count users
        user_counts = Counter()
        for (content,) in rows:
            match = re.match(r"^\[([^\]]+)\]:", content)
            if match:
                user_counts[match.group(1)] += 1
        
        top_users = user_counts.most_common(5)
        
        # Top topics
        cursor.execute(
            "SELECT key, COUNT(*) FROM memory GROUP BY key ORDER BY COUNT(*) DESC LIMIT 5"
        )
        top_topics = cursor.fetchall()
        
        # Top relationships
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
        
        await send_interaction(interaction, "\n".join(lines))
    
    async def reset(self, interaction: discord.Interaction):
        """Clear conversation history."""
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
        
        await interaction.response.send_message(
            "fresh start 🧹 i remember nothing from this chat",
            ephemeral=True,
        )
    
    async def help_command(self, interaction: discord.Interaction):
        """Show all commands."""
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
            "`/tokens` — show your API token usage",
            "`/help` — show this list",
        ]
        await send_interaction(interaction, "\n".join(lines))
