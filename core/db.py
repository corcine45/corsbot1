import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("corsbot.db")

# Use /app for Railway persistent volume, fall back to local for dev
DATA_DIR = (
    Path("/data") if Path("/data").exists() else Path(__file__).resolve().parents[1]
)
DB_PATH = DATA_DIR / "brain.db"
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_RETENTION = 5
SCHEMA_VERSION = 11

IDENTITY_KEYS = {
    "name",
    "age",
    "location",
    "job",
    "display_name",
    "birthday",
    "gender",
    "nationality",
}
TEMPORARY_KEYS = {
    "mood",
    "currently",
    "doing",
    "feeling",
    "status",
    "playing_now",
    "watching_now",
}

_local = threading.local()
_backup_done = False


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def backup_db():
    global _backup_done
    if _backup_done or not DB_PATH.exists():
        return

    ensure_data_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    backup_path = BACKUP_DIR / f"brain-{timestamp}.db.bak"
    try:
        shutil.copy2(DB_PATH, backup_path)
        backups = sorted(BACKUP_DIR.glob("brain-*.db.bak"), key=os.path.getmtime)
        for old_backup in backups[:-BACKUP_RETENTION]:
            old_backup.unlink()
        log.info(f"[db] backup created: {backup_path}")
    except Exception as e:
        log.error(f"[db] backup failed: {e}")
    _backup_done = True


def connect_db():
    ensure_data_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    return conn, cursor


def initialize_schema(cursor):
    cursor.execute("PRAGMA user_version")
    version = cursor.fetchone()[0] or 0

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT,
            key TEXT,
            value TEXT,
            updated_at REAL,
            embedding BLOB,
            memory_type TEXT DEFAULT 'preference',
            reinforcement INTEGER DEFAULT 1,
            PRIMARY KEY (user_id, key)
        )
    """)

    cursor.execute("""
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            user_id TEXT PRIMARY KEY,
            insight TEXT,
            updated_at REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            thread_id TEXT PRIMARY KEY,
            summary TEXT,
            message_count INTEGER DEFAULT 0,
            updated_at REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deferred_instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requester_id TEXT,
            guild_id TEXT,
            channel_id TEXT,
            trigger_type TEXT,
            trigger_target TEXT,
            action TEXT,
            created_at REAL,
            fired INTEGER DEFAULT 0
        )
    """)
    deferred_instruction_cols = [
        row[1]
        for row in cursor.execute("PRAGMA table_info(deferred_instructions)").fetchall()
    ]
    if "trigger_target_id" not in deferred_instruction_cols:
        cursor.execute(
            "ALTER TABLE deferred_instructions ADD COLUMN trigger_target_id TEXT"
        )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            thread_id TEXT,
            reply_snippet TEXT,
            rating TEXT,
            mood TEXT,
            guild_id TEXT,
            timestamp REAL
        )
    """)
    feedback_cols = [
        row[1] for row in cursor.execute("PRAGMA table_info(feedback)").fetchall()
    ]
    if "guild_id" not in feedback_cols:
        cursor.execute("ALTER TABLE feedback ADD COLUMN guild_id TEXT")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            tokens_used INTEGER,
            model TEXT,
            timestamp REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS presence_patterns (
            user_id TEXT,
            pattern_key TEXT,
            kind TEXT,
            subject TEXT,
            count INTEGER DEFAULT 0,
            night_count INTEGER DEFAULT 0,
            social_count INTEGER DEFAULT 0,
            updated_at REAL,
            summary TEXT,
            PRIMARY KEY (user_id, pattern_key)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            guild_id TEXT,
            channel_id TEXT,
            message TEXT,
            fire_at REAL,
            created_at REAL,
            fired INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            last_digest_at REAL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            guild_id TEXT,
            trace_summary TEXT,
            latency_ms REAL,
            route TEXT,
            created_at REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guild_settings (
            guild_id TEXT PRIMARY KEY,
            personality TEXT DEFAULT '',
            memory_isolated INTEGER DEFAULT 0,
            opt_out_channels TEXT DEFAULT '[]',
            updated_at REAL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS music_queues (
            guild_id TEXT PRIMARY KEY,
            now_playing TEXT,
            queue TEXT,
            updated_at REAL DEFAULT 0
        )
    """)

    cols = [row[1] for row in cursor.execute("PRAGMA table_info(memory)").fetchall()]
    if "key" not in cols and "fact" in cols:
        cursor.execute("ALTER TABLE memory RENAME TO memory_old")
        cursor.execute("""
            CREATE TABLE memory (
                user_id TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL,
                embedding BLOB,
                memory_type TEXT DEFAULT 'preference',
                reinforcement INTEGER DEFAULT 1,
                PRIMARY KEY (user_id, key)
            )
        """)
        old_rows = cursor.execute("SELECT user_id, fact FROM memory_old").fetchall()
        for user_id, fact in old_rows:
            if "=" in fact:
                key, value = fact.split("=", 1)
                cursor.execute(
                    "INSERT OR IGNORE INTO memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                    (user_id, key.strip().lower(), value.strip(), time.time()),
                )
        cursor.execute("DROP TABLE memory_old")
        cols = [
            row[1] for row in cursor.execute("PRAGMA table_info(memory)").fetchall()
        ]

    if "embedding" not in cols:
        cursor.execute("ALTER TABLE memory ADD COLUMN embedding BLOB")
        cols.append("embedding")
    if "memory_type" not in cols:
        cursor.execute(
            "ALTER TABLE memory ADD COLUMN memory_type TEXT DEFAULT 'preference'"
        )
        cols.append("memory_type")
    if "reinforcement" not in cols:
        cursor.execute("ALTER TABLE memory ADD COLUMN reinforcement INTEGER DEFAULT 1")
        cols.append("reinforcement")
    if "sensitivity" not in cols:
        cursor.execute(
            "ALTER TABLE memory ADD COLUMN sensitivity TEXT DEFAULT 'normal'"
        )
        cols.append("sensitivity")
    if "confidence" not in cols:
        cursor.execute("ALTER TABLE memory ADD COLUMN confidence REAL DEFAULT 1.0")
        cols.append("confidence")
    if "last_accessed" not in cols:
        cursor.execute("ALTER TABLE memory ADD COLUMN last_accessed REAL DEFAULT 0")
        cols.append("last_accessed")
        # Backfill: set last_accessed = updated_at for existing rows
        cursor.execute(
            "UPDATE memory SET last_accessed = updated_at WHERE last_accessed = 0"
        )

    cursor.execute(
        f"UPDATE memory SET memory_type='identity' WHERE (memory_type IS NULL OR memory_type='') AND LOWER(key) IN ({','.join('?' for _ in IDENTITY_KEYS)})",
        tuple(IDENTITY_KEYS),
    )
    cursor.execute(
        f"UPDATE memory SET memory_type='temporary' WHERE (memory_type IS NULL OR memory_type='') AND LOWER(key) IN ({','.join('?' for _ in TEMPORARY_KEYS)})",
        tuple(TEMPORARY_KEYS),
    )
    cursor.execute(
        "UPDATE memory SET memory_type='preference' WHERE memory_type IS NULL OR memory_type=''"
    )

    if version < SCHEMA_VERSION:
        cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    settings_cols = [
        row[1] for row in cursor.execute("PRAGMA table_info(user_settings)").fetchall()
    ]
    if "last_channel_id" not in settings_cols:
        cursor.execute("ALTER TABLE user_settings ADD COLUMN last_channel_id TEXT")
    if "last_guild_id" not in settings_cols:
        cursor.execute("ALTER TABLE user_settings ADD COLUMN last_guild_id TEXT")
    if "last_message_at" not in settings_cols:
        cursor.execute("ALTER TABLE user_settings ADD COLUMN last_message_at REAL DEFAULT 0")

    # Indexes for performance
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, timestamp)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON memory(user_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tokens_usage_user ON tokens_usage(user_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_presence_patterns_user ON presence_patterns(user_id, updated_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_deferred_instructions_target_id ON deferred_instructions(guild_id, trigger_type, trigger_target_id, fired)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(fired, fire_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_traces_guild ON agent_traces(guild_id, created_at)"
    )


def get_db():
    if not hasattr(_local, "conn"):
        backup_db()
        _local.conn, _local.cursor = connect_db()
        initialize_schema(_local.cursor)
        _local.conn.commit()

    return _local.conn, _local.cursor


def get_thread_id(
    user_id, guild_id=None, channel_id=None, is_dm=False, discord_thread_id=None
):
    if is_dm:
        return f"dm:{user_id}"
    base = f"guild:{guild_id}:channel:{channel_id}"
    if discord_thread_id:
        base += f":thread:{discord_thread_id}"
    return f"{base}:user:{user_id}"


def get_conversation_thread_id(
    user_id, guild_id=None, channel_id=None, is_dm=False, discord_thread_id=None
):
    if is_dm:
        return f"dm:{user_id}"
    base = f"guild:{guild_id}:channel:{channel_id}"
    if discord_thread_id:
        base += f":thread:{discord_thread_id}"
    return base


def get_guild_token_stats(guild_id: int, member_ids: list[int]) -> dict:
    """Approximate guild-wide token usage for known member IDs."""
    if not member_ids:
        return {"total": 0, "today": 0, "requests": 0}
    _, cursor = get_db()
    placeholders = ",".join("?" for _ in member_ids)
    ids = [str(mid) for mid in member_ids]
    cursor.execute(
        f"SELECT COALESCE(SUM(tokens_used), 0) FROM tokens_usage WHERE user_id IN ({placeholders})",
        ids,
    )
    total = cursor.fetchone()[0]
    today_start = time.time() - (24 * 3600)
    cursor.execute(
        f"SELECT COALESCE(SUM(tokens_used), 0) FROM tokens_usage WHERE user_id IN ({placeholders}) AND timestamp > ?",
        ids + [today_start],
    )
    today = cursor.fetchone()[0]
    cursor.execute(
        f"SELECT COUNT(*) FROM tokens_usage WHERE user_id IN ({placeholders})",
        ids,
    )
    requests = cursor.fetchone()[0]
    return {"total": total, "today": today, "requests": requests}


def touch_user_channel(user_id: int, channel_id: int, guild_id: int | None = None):
    conn, cursor = get_db()
    now = time.time()
    cursor.execute(
        """INSERT INTO user_settings (user_id, last_channel_id, last_guild_id, last_message_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               last_channel_id=excluded.last_channel_id,
               last_guild_id=excluded.last_guild_id,
               last_message_at=excluded.last_message_at""",
        (str(user_id), str(channel_id), str(guild_id) if guild_id else None, now),
    )
    conn.commit()


def get_user_last_channel(user_id: int) -> tuple[str | None, str | None]:
    _, cursor = get_db()
    cursor.execute(
        "SELECT last_channel_id, last_guild_id FROM user_settings WHERE user_id=?",
        (str(user_id),),
    )
    row = cursor.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def get_last_message_age(user_id: int) -> float:
    _, cursor = get_db()
    cursor.execute(
        "SELECT last_message_at FROM user_settings WHERE user_id=?",
        (str(user_id),),
    )
    row = cursor.fetchone()
    if row and row[0]:
        return max(0.0, time.time() - row[0])

    cursor.execute(
        "SELECT MAX(timestamp) FROM messages WHERE thread_id LIKE ?",
        (f"%user:{user_id}",),
    )
    ts_row = cursor.fetchone()
    if ts_row and ts_row[0]:
        return max(0.0, time.time() - ts_row[0])
    return 999999.0


def store_message(thread_id, role, content):
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (thread_id, role, content, time.time()),
    )
    conn.commit()


def get_history(thread_id, limit=8):
    _, cursor = get_db()
    cursor.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY timestamp DESC LIMIT ?",
        (thread_id, limit),
    )
    rows = cursor.fetchall()[::-1]
    return [
        {"role": r, "content": c if len(c) <= 900 else c[:899] + "…"} for r, c in rows
    ]


def get_recent_speakers(thread_id, seconds=60, exclude_names=None):
    exclude = {name.lower() for name in (exclude_names or []) if name}
    _, cursor = get_db()
    cursor.execute(
        "SELECT content FROM messages WHERE thread_id=? AND role='user' AND timestamp>=? ORDER BY timestamp DESC",
        (thread_id, time.time() - seconds),
    )

    speakers = []
    seen = set()
    for (content,) in cursor.fetchall():
        match = re.match(r"^\[([^\]]+)\]:", content)
        if not match:
            continue
        name = match.group(1).strip()
        key = name.lower()
        if not name or key in exclude or key in seen:
            continue
        seen.add(key)
        speakers.append(name)
    return speakers


def store_token_usage(user_id, tokens_used, model=""):
    """Record API token usage for a user."""
    conn, cursor = get_db()
    cursor.execute(
        "INSERT INTO tokens_usage (user_id, tokens_used, model, timestamp) VALUES (?, ?, ?, ?)",
        (str(user_id), tokens_used, model, time.time()),
    )
    conn.commit()


def get_token_stats(user_id):
    """Get token usage stats for a user (total and today)."""
    _, cursor = get_db()

    # Total tokens
    cursor.execute(
        "SELECT COALESCE(SUM(tokens_used), 0) FROM tokens_usage WHERE user_id=?",
        (str(user_id),),
    )
    total = cursor.fetchone()[0]

    # Tokens today (last 24 hours)
    now = time.time()
    today_start = now - (24 * 3600)
    cursor.execute(
        "SELECT COALESCE(SUM(tokens_used), 0) FROM tokens_usage WHERE user_id=? AND timestamp > ?",
        (str(user_id), today_start),
    )
    today = cursor.fetchone()[0]

    # Count of requests
    cursor.execute("SELECT COUNT(*) FROM tokens_usage WHERE user_id=?", (str(user_id),))
    requests = cursor.fetchone()[0]

    return {"total": total, "today": today, "requests": requests}
