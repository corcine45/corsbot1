import sqlite3
import threading
import time
import os
import logging
import pickle
import numpy as np
import faiss
from functools import lru_cache
from sentence_transformers import SentenceTransformer

log = logging.getLogger("corsbot.memory")

from .db import get_db, DATA_DIR
from .ai import groq_call

# ---------------- EMBEDDER ---------------- #

print("Loading embedding model...")
embedder = SentenceTransformer("all-mpnet-base-v2")
EMBED_DIM = 768
log.info("Embedding model ready.")

@lru_cache(maxsize=512)
def _embed_vec(text: str) -> np.ndarray:
    return embedder.encode(text, normalize_embeddings=True).astype(np.float32)

def _embed(text: str) -> bytes:
    return _embed_vec(text).tobytes()

# ---------------- FAISS ---------------- #

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "brain.index")
FAISS_MAP_PATH   = os.path.join(DATA_DIR, "brain.index.map")

_faiss_lock = threading.Lock()
_faiss_index: faiss.IndexFlatIP = None
_faiss_map: dict = {}
_faiss_reverse: dict = {}

def _faiss_load():
    global _faiss_index, _faiss_map, _faiss_reverse
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAP_PATH):
        try:
            idx = faiss.read_index(FAISS_INDEX_PATH)
            # If dimension changed (e.g. model upgrade), rebuild
            if idx.d != EMBED_DIM:
                log.warning(f"[faiss] dimension mismatch ({idx.d} vs {EMBED_DIM}) — rebuilding")
                raise ValueError("dimension mismatch")
            _faiss_index = idx
            with open(FAISS_MAP_PATH, "rb") as f:
                _faiss_map = pickle.load(f)
            _faiss_reverse = {v: k for k, v in _faiss_map.items()}
            log.info(f"[faiss] loaded {_faiss_index.ntotal} vectors")
            return
        except Exception as e:
            log.error(f"[faiss] failed to load: {e} — rebuilding")
    _faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    _faiss_map = {}
    _faiss_reverse = {}
    log.info("[faiss] created fresh index")

def _faiss_save():
    faiss.write_index(_faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "wb") as f:
        pickle.dump(_faiss_map, f)

def faiss_upsert(user_id: str, key: str, vec: np.ndarray):
    label = f"{user_id}:{key}"
    with _faiss_lock:
        if label in _faiss_reverse:
            old_id = _faiss_reverse[label]
            del _faiss_map[old_id]
            del _faiss_reverse[label]
        new_id = _faiss_index.ntotal
        _faiss_index.add(vec.reshape(1, -1).astype(np.float32))
        _faiss_map[new_id] = label
        _faiss_reverse[label] = new_id
        _faiss_save()

def faiss_search(user_id: str, query_vec: np.ndarray, top_k: int = 20) -> list:
    with _faiss_lock:
        if _faiss_index.ntotal == 0:
            return []
        k = min(top_k * 4, _faiss_index.ntotal)
        scores, ids = _faiss_index.search(query_vec.reshape(1, -1).astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        label = _faiss_map.get(idx)
        if not label:
            continue
        uid, key = label.split(":", 1)
        if uid == str(user_id):
            results.append((float(score), key))
        if len(results) >= top_k:
            break
    return results

def faiss_rebuild_from_db():
    conn = sqlite3.connect("brain.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id, key, value FROM memory")
        rows = cursor.fetchall()
    except Exception:
        conn.close()
        return
    conn.close()
    missing = [(uid, k, v) for uid, k, v in rows if f"{uid}:{k}" not in _faiss_reverse]
    if not missing:
        return
    log.info(f"[faiss] indexing {len(missing)} missing facts...")
    for uid, key, value in missing:
        vec = _embed_vec(f"{key}: {value}")
        faiss_upsert(uid, key, vec)
    log.info("[faiss] rebuild complete")

# Init on import
_faiss_load()
faiss_rebuild_from_db()

# ---------------- MEMORY TYPES ---------------- #

IDENTITY_KEYS = {"name", "age", "location", "job", "display_name", "birthday", "gender", "nationality"}
TEMPORARY_KEYS = {"mood", "currently", "doing", "feeling", "status", "playing_now", "watching_now"}

DECAY = {
    "identity":   None,
    "preference": 60 * 60 * 24 * 30,
    "temporary":  60 * 60 * 6,
}

def _classify_key(key: str) -> str:
    if key in IDENTITY_KEYS:
        return "identity"
    if key in TEMPORARY_KEYS:
        return "temporary"
    return "preference"


def _resolve_memory_type(key: str, memory_type: str | None) -> str:
    return memory_type or _classify_key(key)


def _decay_score(memory_type: str, updated_at: float, reinforcement: int) -> float:
    half_life = DECAY.get(memory_type)
    if half_life is None:
        return 1.0
    age = time.time() - updated_at
    boost = min(reinforcement, 5) / 5
    decay = 0.5 ** (age / half_life)
    return decay * (0.5 + 0.5 * boost)

# ---------------- EXTRACT / GET / STORE ---------------- #

MEMORY_EXTRACT_EVERY = 1
MEMORY_SIMILARITY_THRESHOLD = 0.35
MAX_MEMORY_FACTS = 6
_msg_counter: dict = {}

def should_extract(user_id: str) -> bool:
    from collections import defaultdict
    _msg_counter.setdefault(user_id, 0)
    _msg_counter[user_id] += 1
    return _msg_counter[user_id] % MEMORY_EXTRACT_EVERY == 0

def extract_memory(user_id, message):
    if len(message.split()) < 5:
        return
    try:
        output = groq_call(
            "llama-3.3-70b-versatile",
            [
                {"role": "system", "content": (
                    "Extract facts about the user from their message. "
                    "Reply in key=value format, one per line. "
                    "Use snake_case keys like: name, age, location, likes, dislikes, favorite_game, job, mood, currently, nickname, title, etc. "
                    "Also capture self-given titles or nicknames — if they say 'call me X', 'I am X', 'I'm the X', 'refer to me as X', store title=X or nickname=X. "
                    "If they say 'call me king of aura', store title=king of aura. "
                    "Only extract clear personal facts explicitly stated. If none, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=80, retries=2, timeout=10,
        )
        if output.strip().upper() == "NONE":
            return

        conn, cursor = get_db()
        now = time.time()
        for line in output.split("\n"):
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip().lower(), value.strip()
            if not key or not value:
                continue
            memory_type = _classify_key(key)
            vec = _embed_vec(f"{key}: {value}")
            cursor.execute("SELECT reinforcement FROM memory WHERE user_id=? AND key=?", (str(user_id), key))
            row = cursor.fetchone()
            reinforcement = (row[0] + 1) if row else 1
            cursor.execute(
                """INSERT INTO memory (user_id, key, value, updated_at, embedding, memory_type, reinforcement)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, key) DO UPDATE SET
                       value=excluded.value, updated_at=excluded.updated_at,
                       embedding=excluded.embedding, memory_type=excluded.memory_type,
                       reinforcement=excluded.reinforcement""",
                (str(user_id), key, value, now, vec.tobytes(), memory_type, reinforcement),
            )
            faiss_upsert(str(user_id), key, vec)
        conn.commit()
    except Exception as e:
        log.error(f"memory extract error: {type(e).__name__}: {e}")

def _expand_query(query: str) -> str:
    """Expand query with key intent words for better semantic matching."""
    # Strip filler and keep the core intent
    filler = {"hey", "corsbot", "can you", "do you", "what is", "tell me", "i want", "please"}
    words = [w for w in query.lower().split() if w not in filler]
    return " ".join(words[:20])  # cap at 20 words

def get_memory(user_id, query: str = "", top_k: int = 8) -> str:
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value, memory_type, updated_at, reinforcement FROM memory WHERE user_id=?",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        return ""

    now = time.time()
    fact_lookup = {}
    for key, value, memory_type, updated_at, reinforcement in rows:
        resolved_type = _resolve_memory_type(key, memory_type)
        fact_lookup[key] = (value, resolved_type, updated_at, reinforcement or 1)

    non_identity = {}
    for key, (value, memory_type, updated_at, reinforcement) in fact_lookup.items():
        half_life = DECAY.get(memory_type)
        if half_life and (now - updated_at) > half_life * 3:
            continue
        non_identity[key] = (value, memory_type, updated_at, reinforcement)

    scored_facts = []
    if query and non_identity:
        expanded = _expand_query(query)
        # Average embeddings of original + expanded query for better coverage
        q1 = _embed_vec(query)
        q2 = _embed_vec(expanded) if expanded != query.lower() else q1
        query_vec = ((q1 + q2) / 2).astype(np.float32)
        # Renormalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        for sim, key in faiss_search(str(user_id), query_vec, top_k=len(non_identity)):
            if key not in non_identity:
                continue
            if sim < MEMORY_SIMILARITY_THRESHOLD:
                continue
            value, memory_type, updated_at, reinforcement = non_identity[key]
            scored_facts.append((sim * _decay_score(memory_type, updated_at, reinforcement), key, value))
    else:
        for key, (value, memory_type, updated_at, reinforcement) in non_identity.items():
            scored_facts.append((_decay_score(memory_type, updated_at, reinforcement), key, value))

    scored_facts.sort(reverse=True)
    result = [f"{k}={v}" for _, k, v in scored_facts[:MAX_MEMORY_FACTS]]
    return "\n".join(result)


def get_memory_with_keys(user_id, query: str = "", top_k: int = 8) -> tuple:
    """Same as get_memory but also returns the list of active keys (for feedback boosting)."""
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value, memory_type, updated_at, reinforcement FROM memory WHERE user_id=?",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        return "", []

    now = time.time()
    fact_lookup = {}
    for key, value, memory_type, updated_at, reinforcement in rows:
        resolved_type = _resolve_memory_type(key, memory_type)
        fact_lookup[key] = (value, resolved_type, updated_at, reinforcement or 1)

    non_identity = {}
    for key, (value, memory_type, updated_at, reinforcement) in fact_lookup.items():
        half_life = DECAY.get(memory_type)
        if half_life and (now - updated_at) > half_life * 3:
            continue
        non_identity[key] = (value, memory_type, updated_at, reinforcement)

    scored_facts = []
    if query and non_identity:
        expanded = _expand_query(query)
        q1 = _embed_vec(query)
        q2 = _embed_vec(expanded) if expanded != query.lower() else q1
        query_vec = ((q1 + q2) / 2).astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        for sim, key in faiss_search(str(user_id), query_vec, top_k=len(non_identity)):
            if key not in non_identity:
                continue
            if sim < MEMORY_SIMILARITY_THRESHOLD:
                continue
            value, memory_type, updated_at, reinforcement = non_identity[key]
            scored_facts.append((sim * _decay_score(memory_type, updated_at, reinforcement), key, value))
    else:
        for key, (value, memory_type, updated_at, reinforcement) in non_identity.items():
            scored_facts.append((_decay_score(memory_type, updated_at, reinforcement), key, value))

    scored_facts.sort(reverse=True)
    top_scored = scored_facts[:MAX_MEMORY_FACTS]
    active_keys = [k for _, k, _ in top_scored]

    result = [f"{k}={v}" for _, k, v in top_scored]
    return "\n".join(result), active_keys

def store_user_name(user_id, display_name, username=None, guild_nick=None):
    """Store all name variants for a user for better identification."""
    conn, cursor = get_db()
    now = time.time()

    # Always store display_name
    vec = _embed_vec(f"display_name: {display_name}")
    cursor.execute(
        """INSERT INTO memory (user_id, key, value, updated_at, embedding, memory_type, reinforcement)
           VALUES (?, 'display_name', ?, ?, ?, 'identity', 1)
           ON CONFLICT(user_id, key) DO UPDATE SET
               value=excluded.value, updated_at=excluded.updated_at,
               embedding=excluded.embedding, memory_type='identity'""",
        (str(user_id), display_name, now, vec.tobytes()),
    )
    faiss_upsert(str(user_id), "display_name", vec)

    # Store username (the @handle) if provided
    if username:
        vec2 = _embed_vec(f"username: {username}")
        cursor.execute(
            """INSERT INTO memory (user_id, key, value, updated_at, embedding, memory_type, reinforcement)
               VALUES (?, 'username', ?, ?, ?, 'identity', 1)
               ON CONFLICT(user_id, key) DO UPDATE SET
                   value=excluded.value, updated_at=excluded.updated_at,
                   embedding=excluded.embedding, memory_type='identity'""",
            (str(user_id), username, now, vec2.tobytes()),
        )
        faiss_upsert(str(user_id), "username", vec2)

    # Store server nickname if different from display name
    if guild_nick and guild_nick != display_name:
        vec3 = _embed_vec(f"server_nickname: {guild_nick}")
        cursor.execute(
            """INSERT INTO memory (user_id, key, value, updated_at, embedding, memory_type, reinforcement)
               VALUES (?, 'server_nickname', ?, ?, ?, 'identity', 1)
               ON CONFLICT(user_id, key) DO UPDATE SET
                   value=excluded.value, updated_at=excluded.updated_at,
                   embedding=excluded.embedding, memory_type='identity'""",
            (str(user_id), guild_nick, now, vec3.tobytes()),
        )
        faiss_upsert(str(user_id), "server_nickname", vec3)

    conn.commit()

# ---------------- RELATIONSHIPS ---------------- #

def extract_relationships(user_id, message):
    # Only bother if message mentions someone by name (has @ or common relationship words)
    lower = message.lower()
    relationship_hints = ("my friend", "my brother", "my sister", "my mom", "my dad", "my girlfriend",
                          "my boyfriend", "my wife", "my husband", "my teammate", "my coworker", "@")
    if not any(hint in lower for hint in relationship_hints):
        return
    if len(message.split()) < 4:
        return
    try:
        output = groq_call(
            "llama-3.3-70b-versatile",
            [
                {"role": "system", "content": (
                    "Extract relationships the user mentions about people in their life. "
                    "Reply in format: name|relation|context, one per line. "
                    "Examples: Mike|friend|we play Valorant together\n"
                    "Only extract explicitly stated relationships. If none, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=100, retries=2, timeout=10,
        )
        if output.strip().upper() == "NONE":
            return

        conn, cursor = get_db()
        now = time.time()
        for line in output.strip().split("\n"):
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            relation = parts[1].strip().lower()
            context = parts[2].strip() if len(parts) > 2 else ""
            if not name or not relation:
                continue
            cursor.execute("SELECT strength FROM relationships WHERE user_id=? AND related_name=?", (str(user_id), name))
            row = cursor.fetchone()
            strength = (row[0] + 1) if row else 1
            cursor.execute(
                """INSERT INTO relationships (user_id, related_name, relation, context, strength, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, related_name) DO UPDATE SET
                       relation=excluded.relation, context=excluded.context,
                       strength=excluded.strength, updated_at=excluded.updated_at""",
                (str(user_id), name, relation, context, strength, now),
            )
        conn.commit()
    except Exception as e:
        log.error(f"relationship extract error: {type(e).__name__}: {e}")

def get_relationships(user_id, top_k: int = 6) -> str:
    _, cursor = get_db()
    cursor.execute(
        "SELECT related_name, relation, context, strength FROM relationships "
        "WHERE user_id=? ORDER BY strength DESC, updated_at DESC LIMIT ?",
        (str(user_id), top_k)
    )
    rows = cursor.fetchall()
    if not rows:
        return ""
    lines = []
    for name, relation, context, strength in rows:
        line = f"{name} ({relation})"
        if context:
            line += f" — {context}"
        lines.append(line)
    return "\n".join(lines)

def search_memory_by_value(query: str, top_k: int = 5) -> str:
    """Search across ALL users' memory for a value matching the query.
    Used for 'who is X' type questions. Returns clear attribution."""
    _, cursor = get_db()
    cursor.execute("SELECT user_id, key, value FROM memory")
    rows = cursor.fetchall()
    if not rows or not query:
        return ""

    query_vec = _embed_vec(query)
    scored = []
    for user_id, key, value in rows:
        fact_text = f"{key}: {value}"
        fact_vec = _embed_vec(fact_text)
        sim = float(np.dot(query_vec, fact_vec))
        if sim > 0.3:
            scored.append((sim, user_id, key, value))

    scored.sort(reverse=True)
    if not scored:
        return ""

    lines = []
    seen_users = set()
    for sim, uid, key, value in scored[:top_k]:
        cursor.execute(
            "SELECT value FROM memory WHERE user_id=? AND key='display_name'",
            (uid,)
        )
        name_row = cursor.fetchone()
        username = name_row[0] if name_row else f"user:{uid}"

        if uid in seen_users:
            continue
        seen_users.add(uid)

        lines.append(f"{username} (user_id:{uid}) declared: {key}={value}")

    return "\n".join(lines)

def _extract_facts_about(user_id: int, message: str) -> str:
    """Extract facts claimed about a mentioned user by someone else.
    Returns the raw claim text for confirmation, or empty string if nothing found."""
    if len(message.split()) < 3:
        return ""
    try:
        output = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Someone is making a claim about another person in this message. "
                    "Extract only the factual claims being made about the mentioned person (not the speaker). "
                    "Reply with a short plain-English summary of what's being claimed about them. "
                    "Example: 'is funny, plays Valorant' or 'is the king of aura'. "
                    "If no clear claims about another person, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=60, retries=1, timeout=8,
        )
        result = output.strip()
        if result.upper() == "NONE" or not result:
            return ""
        return result
    except Exception as e:
        log.error(f"_extract_facts_about error: {e}")
        return ""
