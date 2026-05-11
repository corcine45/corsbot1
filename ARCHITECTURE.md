# Corsbot Architecture Refactoring

## Overview
The bot.py file has been split into a modular architecture for better maintainability, testability, and scalability.

## Directory Structure

```
corsbot/
├── bot.py                 # Entry point (forwards to main.py)
├── main.py               # Bot initialization & orchestration
├── config.py             # Configuration & constants
├── handlers/
│   ├── __init__.py       # MessageHandler - Discord message processing
│   └── commands.py       # CommandsHandler - Slash command definitions
├── models/
│   └── __init__.py       # Discord UI models (Views, Modals)
├── utils/
│   └── __init__.py       # Shared utilities (caching, message sending, etc.)
├── core/                 # AI pipeline (unchanged)
│   ├── agent.py
│   ├── ai.py
│   ├── db.py
│   ├── emotion.py
│   ├── feedback.py
│   ├── memory.py
│   ├── presence.py
│   ├── search.py
│   ├── session.py
│   └── user_state.py
├── brain.index          # Vector database
├── requirements.txt
└── ...
```

## Key Improvements

### 1. **Separation of Concerns**
- **Discord Event Handling** → `handlers/messages.py` (`MessageHandler`)
- **Slash Commands** → `handlers/commands.py` (`CommandsHandler`)
- **Configuration** → `config.py` (all constants & env vars)
- **Utilities** → `utils/` (caching, message formatting, etc.)
- **UI Models** → `models/` (Discord Views, Modals)

### 2. **Testing Benefits**
- Each handler can be unit tested independently
- Config can be mocked for tests
- Utils have no Discord dependencies (pure functions)
- Easier to test AI pipeline in isolation

### 3. **Code Organization**
- **config.py**: ~120 lines (was scattered in bot.py)
- **bot.py**: ~12 lines (was 900+ lines)
- **handlers/messages.py**: ~350 lines (focused on message flow)
- **handlers/commands.py**: ~280 lines (focused on slash commands)
- **utils/__init__.py**: ~250 lines (reusable functions)
- **models/__init__.py**: ~40 lines (Discord UI)

### 4. **Dependency Flow**
```
bot.py (entry)
  └── main.py (setup)
       ├── handlers/messages.py (MessageHandler)
       ├── handlers/commands.py (CommandsHandler)
       ├── config.py (constants)
       ├── utils/ (utilities)
       ├── models/ (UI)
       └── core/ (AI pipeline) [unchanged]
```

## Migration from Old Code

### Old Pattern
```python
# bot.py (900+ lines)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HISTORY_LIMIT = 30
# ... lots of config
_RESPONSE_CACHE = {}
_user_cooldowns = {}

@client.event
async def on_message(message):
    # 400+ lines of logic

@client.tree.command(...)
async def slash_memory(...):
    # 20 lines per command × 10 commands = 200 lines
```

### New Pattern
```python
# config.py (centralized)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HISTORY_LIMIT = 30
QUICK_REPLIES = {...}

# handlers/messages.py (focused)
class MessageHandler:
    async def handle(self, message):
        # Clear, testable logic

# handlers/commands.py (organized)
class CommandsHandler:
    async def memory(self, interaction):  # 30 lines
    async def rate(self, interaction):    # 20 lines
    # etc.

# main.py (orchestration)
bot = CorsBot(executor, response_cache)
bot.message_handler = MessageHandler(...)
bot.commands_handler = CommandsHandler(...)
```

## Running the Bot

```bash
# Still works the same
python bot.py

# Or directly
python main.py
```

## Testing Examples

### Test MessageHandler
```python
from handlers import MessageHandler
from utils import ResponseCache

# Create handler without Discord client
handler = MessageHandler(mock_client, executor, config, quick_replies)
await handler.handle(mock_message)
```

### Test Commands
```python
from handlers.commands import CommandsHandler

handler = CommandsHandler(mock_client, executor)
await handler.memory(mock_interaction)
```

### Test Utils
```python
from utils import build_response_cache_key, ResponseCache

key = build_response_cache_key("thread", "message", "memory")
cache = ResponseCache(ttl_seconds=300)
```

## Future Improvements

1. **Async Database** - Switch from sync SQLite to async DB
2. **Dependency Injection** - Use a DI container for cleaner setup
3. **Event Bus** - Decouple handlers with event publishing
4. **Metrics** - Add Prometheus metrics per handler
5. **Rate Limiting** - Move to a service layer
6. **Middleware** - Add middleware for auth, logging, etc.

## Discord-Specific vs Core Logic

### Discord-Specific (to move out of core)
- Message formatting with Discord mentions
- Emoji reactions
- Embeds/Rich messages
- GIF reactions

### Core Logic (stays in core/)
- AI generation (groq_call)
- Memory extraction
- Emotion classification
- Session state management

This separation makes it easier to:
- Deploy without Discord (CLI, API, etc.)
- Test AI logic independently
- Swap Discord with another platform

---

**Status**: ✅ Refactoring complete
**Files Created**: config.py, main.py, handlers/, models/, utils/
**Files Modified**: bot.py (simplified to 12 lines)
**Backward Compatibility**: ✅ (bot.py still works as entry point)
