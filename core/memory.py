import sqlite3
import threading
import time
import os
import pickle
import numpy as np
import faiss
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from .db import get_db, DATA_DIR
from .ai import groq_call

# ---------------- EMBEDDER ---------------- #

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384
print("Embedding model ready.")

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
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_MAP_PATH, "rb") as f:
                _faiss_map = pickle.load(f)
            _faiss_reverse = {v: k for k, v in _faiss_map.items()}
            print(f"[faiss] loaded {_faiss_index.ntotal} vectors")
            return
        except Exception as e:
            print(f"[faiss] failed to load: {e} — rebuilding")
    _faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    _faiss_map = {}
    _faiss_reverse = {}
    print("[faiss] created fresh index")

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
    print(f"[faiss] indexing {len(missing)} missing facts...")
    for uid, key, value in missing:
        vec = _embed_vec(f"{key}: {value}")
        faiss_upsert(uid, key, vec)
    print("[faiss] rebuild complete")

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

MEMORY_EXTRACT_EVERY = 3
MEMORY_SIMILARITY_THRESHOLD = 0.80
MAX_MEMORY_FACTS = 3
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
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Extract facts about the user from their message. "
                    "Reply in key=value format, one per line. "
                    "Use snake_case keys like: name, age, location, likes, dislikes, favorite_game, job, mood, currently, etc. "
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
        print(f"[memory extract error] {e}")

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
        query_vec = _embed_vec(query)
        for sim, key in faiss_search(str(user_id), query_vec, top_k=len(non_identity)):
            if key not in non_identity:
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
        query_vec = _embed_vec(query)
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

def store_user_name(user_id, display_name):
    conn, cursor = get_db()
    vec = _embed_vec(f"display_name: {display_name}")
    cursor.execute(
        """INSERT INTO memory (user_id, key, value, updated_at, embedding, memory_type, reinforcement)
           VALUES (?, 'display_name', ?, ?, ?, 'identity', 1)
           ON CONFLICT(user_id, key) DO UPDATE SET
               value=excluded.value, updated_at=excluded.updated_at,
               embedding=excluded.embedding, memory_type='identity'""",
        (str(user_id), display_name, time.time(), vec.tobytes()),
    )
    conn.commit()
    faiss_upsert(str(user_id), "display_name", vec)

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
            "llama-3.1-8b-instant",
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
        print(f"[relationship extract error] {e}")

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
