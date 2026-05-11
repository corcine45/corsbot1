"""
Corsbot - Discord AI Bot

Entry point. Run this to start the bot.

For development and understanding the architecture, see:
- main.py: Bot initialization and core setup
- handlers/: Message and command handlers
- config.py: Configuration and constants
- core/: AI, memory, and session management
"""

from main import create_bot
import config

if __name__ == "__main__":
    bot = create_bot()
    bot.run(config.DISCORD_TOKEN)
