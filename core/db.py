import sqlite3
import threading
import time
import os

# Use /app for Railway persistent volume, fall back to local for dev
DATA_DIR = "/app" if os.path.exists("/app") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DATA_DIR, "brain.db")

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.cursor = _local.conn.cursor()

        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)

        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL,
                PRIMARY KEY (user_id, key)
            )
        """)

        # Migrate old schema (user_id, fact) -> (user_id, key, value, updated_at)
        cols = [row[1] for row in _local.cursor.execute("PRAGMA table_info(memory)").fetchall()]
        if "key" not in cols:
            _local.cursor.execute("ALTER TABLE memory RENAME TO memory_old")
            _local.cursor.execute("""
                CREATE TABLE memory (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    updated_at REAL,
                    PRIMARY KEY (user_id, key)
                )
            """)
            old_rows = _local.cursor.execute("SELECT user_id, fact FROM memory_old").fetchall()
            for user_id, fact in old_rows:
                if "=" in fact:
                    k, v = fact.split("=", 1)
                    _local.cursor.execute(
                        "INSERT OR IGNORE INTO memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                        (user_id, k.strip().lower(), v.strip(), time.time())
                    )
            _local.cursor.execute("DROP TABLE memory_old")

        # Add missing columns
        cols = [row[1] for row in _local.cursor.execute("PRAGMA table_info(memory)").fetchall()]
        if "embedding" not in cols:
            _local.cursor.execute("ALTER TABLE memory ADD COLUMN embedding BLOB")
        if "memory_type" not in cols:
            _local.cursor.execute("ALTER TABLE memory ADD COLUMN memory_type TEXT DEFAULT 'preference'")
        if "reinforcement" not in cols:
            _local.cursor.execute("ALTER TABLE memory ADD COLUMN reinforcement INTEGER DEFAULT 1")

        # Relationships table
        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                user_id TEXT,
                related_name TEXT,
                relation TEXT,
                context TEXT,
                strength INTEGER DEFAULT 1,
                updated_at REAL,
                PRIMARY KEY (user_id, related_name)
            )
        """)

        # Feedback table
        _local.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                thread_id TEXT,
                reply_snippet TEXT,
                rating TEXT,
                mood TEXT,
                timestamp REAL
            )
        """)

        _local.conn.commit()

    return _local.conn, _local.cursor


def get_thread_id(user_id, guild_id=None, channel_id=None, is_dm=False):
    return f"dm:{user_id}" if is_dm else f"guild:{guild_id}"


def store_message(thread_id, role, content):
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (thread_id, role, content, time.time()),
    )
    conn.commit()


def get_history(thread_id, limit=10):
    _, cursor = get_db()
    cursor.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY timestamp DESC LIMIT ?",
        (thread_id, limit),
    )
    rows = cursor.fetchall()[::-1]
    return [{"role": r, "content": c} for r, c in rows]
