"""
Discord UI models (buttons, views, etc.)
"""

import asyncio
import discord
from typing import Callable


class DenialConfirmView(discord.ui.View):
    """Confirmation view for deleting memories."""
    
    def __init__(self, user_id: int, facts: list, delete_fn: Callable):
        super().__init__(timeout=60)
        self.user_id = user_id
        self.facts = facts  # list of (key, value)
        self.delete_fn = delete_fn  # async function to delete facts

    @discord.ui.button(label="Yeah delete it", style=discord.ButtonStyle.red)
    async def confirm_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not for you.", ephemeral=True)
            return
        
        loop = asyncio.get_running_loop()
        for key, value in self.facts:
            await loop.run_in_executor(None, self.delete_fn, self.user_id, key)
        
        deleted = ", ".join(f"`{v}`" for _, v in self.facts)
        await interaction.response.send_message(
            f"Gone. Won't remember {deleted} about you anymore.",
            ephemeral=False
        )
        self.stop()

    @discord.ui.button(label="Nah keep it", style=discord.ButtonStyle.grey)
    async def cancel_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Not for you.", ephemeral=True)
            return
        await interaction.response.send_message("Aight, keeping it.", ephemeral=False)
        self.stop()
