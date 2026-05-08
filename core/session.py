
import time
from collections import defaultdict

SESSION_TIMEOUT = 1800
MESSAGE_WINDOW = 4

_sessions = defaultdict(dict)

def add_message(user_id: int, content: str):
    state = _sessions[user_id]
    state.setdefault("messages", [])
    state["messages"].append(content)
    state["messages"] = state["messages"][-MESSAGE_WINDOW:]
    state["last_seen"] = time.time()

def should_refresh(user_id: int) -> bool:
    state = _sessions.get(user_id, {})
    return len(state.get("messages", [])) >= MESSAGE_WINDOW

def set_context(user_id: int, context: str):
    _sessions[user_id]["context"] = context
    _sessions[user_id]["last_seen"] = time.time()

def get_context(user_id: int) -> str:
    state = _sessions.get(user_id)
    if not state:
        return ""
    if time.time() - state.get("last_seen", 0) > SESSION_TIMEOUT:
        _sessions.pop(user_id, None)
        return ""
    return state.get("context", "")

def get_recent_messages(user_id: int):
    return _sessions.get(user_id, {}).get("messages", [])
