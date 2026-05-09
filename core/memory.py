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

log.info("Loading embedding model...")
embedder = SentenceTransformer("all-mpnet-base-v2")
EMBED_DIM = 768
log.info("Embedding model ready.")

@lru_cache(maxsize=512)
def _embed_vec(text: str) -> np.ndarray:
    return embedder.encode(text, normalize_embeddings=True).astype(np.float32)

def _embed(text: str) -> bytes:
    return _embed_vec(text).tobytes()

# ---------------- FAISS ---------------- #

# ---------------- FAISS ---------------- #
#
# Index choice: IndexHNSWFlat
#   - Approximate nearest-neighbor via navigable small-world graph
#   - No ghost vectors after compaction (full rebuild produces a clean graph)
#   - Faster search than IndexFlatIP at scale (sub-linear vs linear)
#   - Tradeoff: slightly approximate results, but at our vector counts (< 100k)
#     the accuracy difference is negligible
#
# Ghost vector problem with IndexFlatIP:
#   FAISS cannot remove individual vectors from a flat index. The old approach
#   deleted the label from the map but left the raw vector in the index, causing
#   ghost hits during search. HNSW + periodic full rebuild eliminates this.
#
# M=32: number of neighbors per node in the HNSW graph.
#   Higher M → better recall, more memory. 32 is a good default for 768-dim.
# efConstruction=200: search depth during index build. Higher → better graph quality.
# efSearch=64: search depth at query time. Higher → better recall, slower search.

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "brain.index")
FAISS_MAP_PATH   = os.path.join(DATA_DIR, "brain.index.map")

HNSW_M               = 32    # neighbors per node
HNSW_EF_CONSTRUCTION = 200   # build-time search depth
HNSW_EF_SEARCH       = 64    # query-time search depth

_faiss_lock = threading.Lock()
_faiss_index: faiss.IndexHNSWFlat = None
_faiss_map: dict = {}       # faiss_id → "user_id:key"
_faiss_reverse: dict = {}   # "user_id:key" → faiss_id
_dirty = False              # True when in-memory index differs from disk


def _make_hnsw() -> faiss.IndexHNSWFlat:
    """Create a fresh HNSW index with consistent parameters."""
    idx = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    idx.hnsw.efSearch = HNSW_EF_SEARCH
    return idx


def _faiss_load():
    global _faiss_index, _faiss_map, _faiss_reverse
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAP_PATH):
        try:
            idx = faiss.read_index(FAISS_INDEX_PATH)
            if idx.d != EMBED_DIM:
                log.warning(f"[faiss] dimension mismatch ({idx.d} vs {EMBED_DIM}) — rebuilding")
                raise ValueError("dimension mismatch")
            # Migrate: if loaded index is the old FlatIP type, force a rebuild
            if not isinstance(idx, faiss.IndexHNSWFlat):
                log.info("[faiss] migrating from IndexFlatIP → IndexHNSWFlat")
                raise ValueError("index type migration")
            # Restore efSearch (not persisted by faiss.write_index for HNSW)
            idx.hnsw.efSearch = HNSW_EF_SEARCH
            _faiss_index = idx
            with open(FAISS_MAP_PATH, "rb") as f:
                _faiss_map = pickle.load(f)
            _faiss_reverse = {v: k for k, v in _faiss_map.items()}
            log.info(f"[faiss] loaded {_faiss_index.ntotal} vectors (HNSW)")
            return
        except Exception as e:
            log.error(f"[faiss] failed to load: {e} — rebuilding")
    _faiss_index = _make_hnsw()
    _faiss_map = {}
    _faiss_reverse = {}
    log.info("[faiss] created fresh HNSW index")


def _faiss_save():
    """Persist index and map to disk. Call only when _dirty is True."""
    faiss.write_index(_faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "wb") as f:
        pickle.dump(_faiss_map, f)


def faiss_compact():
    """
    Rebuild the HNSW index from scratch using only live vectors from the DB.

    This eliminates ghost vectors (vectors whose labels were removed from the map
    but whose raw data still occupies slots in the index). With HNSW, a full
    rebuild also produces a clean graph with no dangling edges.

    DB access is batched into a single query instead of one connection per vector.
    """
    global _faiss_index, _faiss_map, _faiss_reverse, _dirty

    with _faiss_lock:
        live_labels = list(_faiss_map.values())
        if not live_labels:
            return

        # Batch-fetch all live facts in one query
        db_path = str(DATA_DIR / "brain.db") if hasattr(DATA_DIR, "__truediv__") else os.path.join(str(DATA_DIR), "brain.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Build lookup: (user_id, key) → value
        cursor.execute("SELECT user_id, key, value FROM memory")
        db_rows = {(r[0], r[1]): r[2] for r in cursor.fetchall()}
        conn.close()

        new_index = _make_hnsw()
        new_map = {}
        new_reverse = {}
        skipped = 0

        for label in live_labels:
            uid, key = label.split(":", 1)
            value = db_rows.get((uid, key))
            if value is None:
                skipped += 1
                continue  # deleted from DB — drop the ghost
            vec = _embed_vec(f"{key}: {value}")
            new_id = new_index.ntotal
            new_index.add(vec.reshape(1, -1).astype(np.float32))
            new_map[new_id] = label
            new_reverse[label] = new_id

        _faiss_index = new_index
        _faiss_map = new_map
        _faiss_reverse = new_reverse
        _faiss_save()
        _dirty = False
        log.info(f"[faiss] compacted: {len(live_labels)} labels → {new_index.ntotal} vectors (dropped {skipped} ghosts)")


_upsert_count = 0
_COMPACT_EVERY = 100
_SAVE_EVERY    = 10   # flush to disk every N upserts, not every single one


def faiss_upsert(user_id: str, key: str, vec: np.ndarray):
    """
    Add or update a vector in the index.

    HNSW doesn't support in-place updates, so we:
    1. Remove the old label from the map (the old vector becomes a ghost)
    2. Add the new vector and point the label at it
    3. Periodic compaction (every _COMPACT_EVERY upserts) rebuilds the index
       cleanly, eliminating all accumulated ghosts at once.
    """
    global _upsert_count, _dirty

    label = f"{user_id}:{key}"
    with _faiss_lock:
        # Remove old label from map (vector stays in index as ghost until compaction)
        if label in _faiss_reverse:
            old_id = _faiss_reverse[label]
            del _faiss_map[old_id]
            del _faiss_reverse[label]

        new_id = _faiss_index.ntotal
        _faiss_index.add(vec.reshape(1, -1).astype(np.float32))
        _faiss_map[new_id] = label
        _faiss_reverse[label] = new_id
        _dirty = True

        _upsert_count += 1
        # Flush to disk periodically, not on every write
        if _upsert_count % _SAVE_EVERY == 0:
            _faiss_save()
            _dirty = False

    # Compact outside the lock to avoid holding it during the full rebuild
    if _upsert_count % _COMPACT_EVERY == 0:
        log.info(f"[faiss] auto-compacting after {_upsert_count} upserts")
        faiss_compact()


def faiss_search(user_id: str, query_vec: np.ndarray, top_k: int = 20) -> list:
    """
    Search for the top-k most similar vectors for a given user.
    Oversamples by 4x to account for ghost vectors and cross-user results,
    then filters down to the requested user's live labels.
    """
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
            continue  # ghost vector — skip
        uid, key = label.split(":", 1)
        if uid == str(user_id):
            results.append((float(score), key))
        if len(results) >= top_k:
            break
    return results


def faiss_rebuild_from_db():
    """
    Index any facts in the DB that are missing from the FAISS index.
    Uses DATA_DIR for the DB path (fixes the hardcoded 'brain.db' bug).
    Batches all missing facts into a single pass.
    """
    db_path = str(DATA_DIR / "brain.db") if hasattr(DATA_DIR, "__truediv__") else os.path.join(str(DATA_DIR), "brain.db")
    conn = sqlite3.connect(db_path)
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

MEMORY_EXTRACT_EVERY = 3
MEMORY_SIMILARITY_THRESHOLD = 0.48
MAX_MEMORY_FACTS = 6
DEDUP_SIMILARITY_THRESHOLD = 0.92  # facts this similar are considered duplicates
_msg_counter: dict = {}

def should_extract(user_id: str) -> bool:
    from collections import defaultdict
    _msg_counter.setdefault(user_id, 0)
    _msg_counter[user_id] += 1
    return _msg_counter[user_id] % MEMORY_EXTRACT_EVERY == 0

def check_and_delete_denied_facts(user_id, message: str) -> list:
    """If user denies a stored fact, return list of matching (key, value) to confirm deletion.
    Does NOT delete immediately — caller should ask for confirmation first."""
    denial_patterns = [
        r"i'?m not (.+)",
        r"i am not (.+)",
        r"i don'?t (.+)",
        r"i do not (.+)",
        r"i never (.+)",
        r"i'?m no longer (.+)",
        r"i'?m not a (.+)",
        r"i'?m not the (.+)",
        r"that'?s not true",
        r"that is not true",
        r"that'?s false",
        r"that is false",
        r"that'?s wrong",
        r"that is wrong",
        r"delete that",
        r"forget that",
        r"that'?s not me",
        r"that is not me",
    ]
    import re as _re
    lower = message.lower().strip()

    denied_phrase = None
    for pattern in denial_patterns:
        match = _re.search(pattern, lower)
        if match:
            denied_phrase = match.group(1).strip() if match.lastindex else lower
            break

    if not denied_phrase:
        return []

    _, cursor = get_db()
    cursor.execute("SELECT key, value FROM memory WHERE user_id=?", (str(user_id),))
    rows = cursor.fetchall()
    if not rows:
        return []

    denied_vec = _embed_vec(denied_phrase)
    matches = []
    for key, value in rows:
        # Skip identity keys like display_name, username
        if key in ("display_name", "username", "server_nickname"):
            continue
        fact_vec = _embed_vec(f"{key}: {value}")
        sim = float(np.dot(denied_vec, fact_vec))
        if sim > 0.65:
            matches.append((key, value))

    return matches

def delete_denied_fact(user_id, key: str):
    """Delete a fact and store a denial record so it won't be re-extracted."""
    conn, cursor = get_db()
    cursor.execute("DELETE FROM memory WHERE user_id=? AND key=?", (str(user_id), key))
    # Store the denial as a special fact so extraction won't re-add it
    denial_key = f"denied_{key}"
    cursor.execute(
        """INSERT INTO memory (user_id, key, value, updated_at, memory_type, reinforcement)
           VALUES (?, ?, 'denied', ?, 'identity', 99)
           ON CONFLICT(user_id, key) DO UPDATE SET value='denied', updated_at=excluded.updated_at""",
        (str(user_id), denial_key, time.time())
    )
    conn.commit()
    log.info(f"[memory] deleted and locked denied fact: user={user_id} key={key}")

# ---------------- DETERMINISTIC EXTRACTION ---------------- #
# Regex patterns for facts that are always stated explicitly and unambiguously.
# These run BEFORE the AI call — zero hallucination risk, zero cost.

import re as _re

# Each entry: (key, compiled_pattern, value_group_index)
# The pattern must capture the value in a named group called `val`.
_DETERMINISTIC_PATTERNS: list[tuple[str, _re.Pattern]] = [
    # Identity
    ("name",        _re.compile(r"\bmy name is (?P<val>[a-z][a-z\s\-']{0,30})", _re.I)),
    ("name",        _re.compile(r"\bcall me (?P<val>[a-z][a-z\s\-']{0,30})", _re.I)),
    ("name",        _re.compile(r"\bi(?:'m| am) (?P<val>[A-Z][a-z]{1,20})(?:\s|$|,|\.)", 0)),  # "I'm John"
    ("age",         _re.compile(r"\bi(?:'m| am) (?P<val>\d{1,2})\s*(?:years?\s*old|yo\b)", _re.I)),
    ("age",         _re.compile(r"\bmy age is (?P<val>\d{1,2})\b", _re.I)),
    ("age",         _re.compile(r"\bturned (?P<val>\d{1,2})\b", _re.I)),
    ("birthday",    _re.compile(r"\bmy birthday is (?P<val>[a-z0-9 ,/\-]+)", _re.I)),
    ("birthday",    _re.compile(r"\bi was born (?:on )?(?P<val>[a-z0-9 ,/\-]+)", _re.I)),
    ("gender",      _re.compile(r"\bi(?:'m| am) (?:a\s+)?(?P<val>male|female|non.?binary|trans|guy|girl|boy|woman|man)\b", _re.I)),
    ("location",    _re.compile(r"\bi(?:'m| am) from (?P<val>[a-z][a-z\s,]{2,40})", _re.I)),
    ("location",    _re.compile(r"\bi live in (?P<val>[a-z][a-z\s,]{2,40})", _re.I)),
    ("location",    _re.compile(r"\bfrom (?P<val>[a-z][a-z\s,]{2,30})(?:\s|$|,|\.)", _re.I)),
    ("nationality", _re.compile(r"\bi(?:'m| am) (?P<val>filipino|american|british|canadian|australian|japanese|korean|chinese|indian|mexican|brazilian|german|french|spanish|italian|russian|thai|vietnamese|indonesian|malaysian|singaporean)\b", _re.I)),
    ("job",         _re.compile(r"\bi(?:'m| am) (?:a\s+)?(?P<val>student|developer|engineer|designer|teacher|doctor|nurse|lawyer|manager|artist|musician|writer|gamer|streamer|freelancer|programmer|chef|driver|soldier|officer)\b", _re.I)),
    ("job",         _re.compile(r"\bi work (?:as (?:a\s+)?|at )(?P<val>[a-z][a-z\s]{2,30})", _re.I)),
    ("job",         _re.compile(r"\bmy job is (?P<val>[a-z][a-z\s]{2,30})", _re.I)),
    # Preferences
    ("favorite_game",   _re.compile(r"\bmy (?:fav(?:ou?rite)?\s+)?game is (?P<val>[a-z][a-z0-9\s:]{1,40})", _re.I)),
    ("favorite_game",   _re.compile(r"\bi (?:love|main|play) (?P<val>[a-z][a-z0-9\s:]{1,30})\s+(?:a lot|all day|everyday|daily|rn|right now)", _re.I)),
    ("favorite_music",  _re.compile(r"\bmy (?:fav(?:ou?rite)?\s+)?(?:music|song|artist|band) is (?P<val>[a-z][a-z0-9\s,&]{1,40})", _re.I)),
    ("favorite_food",   _re.compile(r"\bmy (?:fav(?:ou?rite)?\s+)?food is (?P<val>[a-z][a-z\s]{1,30})", _re.I)),
    ("favorite_show",   _re.compile(r"\bmy (?:fav(?:ou?rite)?\s+)?(?:show|anime|series) is (?P<val>[a-z][a-z0-9\s:]{1,40})", _re.I)),
    # Temporary state
    ("mood",        _re.compile(r"\bi(?:'m| am) (?:feeling\s+)?(?P<val>happy|sad|angry|bored|tired|excited|stressed|anxious|depressed|lonely|hyped|nervous|frustrated|mad|upset|fine|okay|good|great|terrible|awful)\b", _re.I)),
    ("currently",   _re.compile(r"\bi(?:'m| am) (?:currently\s+)?(?P<val>playing|watching|studying|working|gaming|grinding|chilling|sleeping|eating|coding|streaming)\b", _re.I)),
    ("playing_now", _re.compile(r"\b(?:playing|grinding)\s+(?P<val>[a-z][a-z0-9\s:]{1,30})\s+(?:rn|right now|atm|at the moment)", _re.I)),
    ("watching_now",_re.compile(r"\bwatching\s+(?P<val>[a-z][a-z0-9\s:]{1,30})\s+(?:rn|right now|atm|at the moment)", _re.I)),
    # Titles / nicknames
    ("title",       _re.compile(r"\bi(?:'m| am) the (?P<val>[a-z][a-z\s]{2,40})", _re.I)),
    ("title",       _re.compile(r"\brefer to me as (?P<val>[a-z][a-z\s]{1,40})", _re.I)),
    ("nickname",    _re.compile(r"\bpeople call me (?P<val>[a-z][a-z\s\-']{1,30})", _re.I)),
    ("nickname",    _re.compile(r"\bmy (?:nickname|alias) is (?P<val>[a-z][a-z\s\-']{1,30})", _re.I)),
]

# Keys that the deterministic pass should NOT overwrite if already set with high reinforcement
_DETERMINISTIC_PROTECTED = {"display_name", "username", "server_nickname"}

# Keys where the deterministic pass should skip AI extraction (already handled)
_DETERMINISTIC_COVERED = {
    "name", "age", "birthday", "gender", "location", "nationality",
    "job", "mood", "currently", "playing_now", "watching_now",
    "title", "nickname",
}


def _extract_deterministic(message: str) -> dict[str, str]:
    """
    Run regex patterns against the message and return a dict of {key: value}.
    Zero AI calls. Zero hallucinations.
    """
    found: dict[str, str] = {}
    for key, pattern in _DETERMINISTIC_PATTERNS:
        if key in found:
            continue  # first match wins per key
        m = pattern.search(message)
        if not m:
            continue
        # Handle patterns with named group `val` or fallback to group `val2`
        try:
            val = m.group("val")
        except IndexError:
            continue
        if val is None:
            try:
                val = m.group("val2")
            except IndexError:
                continue
        if val:
            val = val.strip().rstrip(".,!?")
            # Sanity: skip if value is suspiciously long or empty
            if 1 <= len(val) <= 60:
                found[key] = val
    return found


def _store_facts(user_id, facts: dict[str, str], conn, cursor, now: float):
    """
    Shared fact-storage logic used by both extraction passes.
    Handles dedup, reinforcement, embedding, and FAISS upsert.
    """
    for key, value in facts.items():
        if not key or not value:
            continue
        # Skip denied facts
        denial_key = f"denied_{key}"
        cursor.execute("SELECT 1 FROM memory WHERE user_id=? AND key=?", (str(user_id), denial_key))
        if cursor.fetchone():
            continue

        memory_type = _classify_key(key)
        vec = _embed_vec(f"{key}: {value}")

        # Deduplication: check if a very similar fact already exists under a different key
        existing_similar = None
        for sim, existing_key in faiss_search(str(user_id), vec, top_k=3):
            if sim >= DEDUP_SIMILARITY_THRESHOLD and existing_key != key:
                existing_similar = existing_key
                break

        if existing_similar:
            log.info(f"[memory] dedup: '{key}={value}' similar to existing '{existing_similar}', updating")
            cursor.execute(
                "UPDATE memory SET value=?, updated_at=?, reinforcement=reinforcement+1 WHERE user_id=? AND key=?",
                (value, now, str(user_id), existing_similar)
            )
            updated_vec = _embed_vec(f"{existing_similar}: {value}")
            cursor.execute(
                "UPDATE memory SET embedding=? WHERE user_id=? AND key=?",
                (updated_vec.tobytes(), str(user_id), existing_similar)
            )
            faiss_upsert(str(user_id), existing_similar, updated_vec)
            continue

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


def extract_memory(user_id, message):
    """
    Two-pass memory extraction:

    Pass 1 — Deterministic (regex):
        Catches explicit, unambiguous facts like "my name is", "I'm 19",
        "I live in Manila". Zero AI calls, zero hallucination risk.

    Pass 2 — AI (llama-3.1-8b-instant):
        Handles nuanced facts the regex can't catch: personality traits,
        preferences, opinions, self-given titles.
        Only runs if the message is long enough, and skips keys already
        covered by the deterministic pass.
    """
    if len(message.split()) < 3:
        return

    conn, cursor = get_db()
    now = time.time()

    # ── Pass 1: Deterministic ────────────────────────────────────────────
    det_facts = _extract_deterministic(message)
    if det_facts:
        log.debug(f"[memory] deterministic extracted: {det_facts}")
        _store_facts(user_id, det_facts, conn, cursor, now)
        conn.commit()

    # ── Pass 2: AI extraction ────────────────────────────────────────────
    if len(message.split()) < 5:
        return

    skip_keys = _DETERMINISTIC_COVERED | set(det_facts.keys())
    skip_hint = (
        f"Do NOT extract these keys (already handled): {', '.join(sorted(skip_keys))}. "
        if skip_keys else ""
    )

    try:
        output = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Extract nuanced personal facts about the user from their message. "
                    "Focus on: personality traits, preferences, opinions, hobbies, "
                    "self-given titles or nicknames, and things they like/dislike. "
                    f"{skip_hint}"
                    "Reply in key=value format, one per line. "
                    "Use snake_case keys like: likes, dislikes, favorite_game, hobby, "
                    "personality, opinion_on, title, nickname, etc. "
                    "Only extract facts clearly stated or strongly implied. "
                    "If none, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=80, retries=2, timeout=10,
        )
        if output.strip().upper() == "NONE":
            return

        ai_facts: dict[str, str] = {}
        for line in output.split("\n"):
            line = line.strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().lower(), value.strip()
            if not key or not value:
                continue
            if key in det_facts:
                continue  # don't let AI overwrite deterministic results
            ai_facts[key] = value

        if ai_facts:
            log.debug(f"[memory] AI extracted: {ai_facts}")
            _store_facts(user_id, ai_facts, conn, cursor, now)
            conn.commit()

    except Exception as e:
        log.error(f"memory AI extract error: {type(e).__name__}: {e}")

# ---------------- MEMORY SENSITIVITY ---------------- #
#
# Sensitive facts are stored normally but filtered at retrieval time.
# They only surface when the current query is semantically close to the topic —
# i.e. the user is actively talking about it, not just chatting.
#
# Three tiers:
#   BLOCKED  — never surface automatically (trauma, abuse, self-harm, private confessions)
#   HIGH     — only surface when query similarity > HIGH_SENSITIVITY_THRESHOLD
#   NORMAL   — standard retrieval (default)
#
# Key-based classification is fast and deterministic.
# Value-based classification catches things like "trauma=car accident" stored under
# a generic key.

# Keys that are always blocked from automatic surfacing
_BLOCKED_KEYS: frozenset[str] = frozenset({
    "trauma", "abuse", "assault", "rape", "suicide", "self_harm", "self_harm_history",
    "mental_illness", "diagnosis", "medication", "therapy", "therapist",
    "addiction", "drug_use", "alcohol_problem", "eating_disorder",
    "confession", "secret", "private", "dont_share", "do_not_share",
    "sexual_history", "sexual_preference", "kink", "fetish",
    "financial_debt", "bankruptcy", "criminal_record", "arrest",
    "family_abuse", "domestic_violence", "cheating", "affair",
})

# Keys that are high-sensitivity — only surface when directly relevant
_HIGH_SENSITIVITY_KEYS: frozenset[str] = frozenset({
    "depression", "anxiety_disorder", "mental_health", "grief", "loss",
    "breakup", "divorce", "relationship_status", "ex", "ex_girlfriend",
    "ex_boyfriend", "ex_wife", "ex_husband", "heartbreak",
    "death", "died", "passed_away", "funeral", "mourning",
    "fired", "job_loss", "unemployed", "financial_struggle",
    "fight_with", "argument_with", "conflict_with",
    "insecurity", "fear", "phobia", "nightmare",
    "loneliness", "isolation", "bullying", "harassment",
})

# Value substrings that flag a fact as high-sensitivity regardless of key
_SENSITIVE_VALUE_SIGNALS: tuple[str, ...] = (
    "died", "dead", "passed away", "suicide", "self harm", "self-harm",
    "abuse", "assault", "trauma", "raped", "molested",
    "depressed", "depression", "anxiety", "panic attack",
    "broke up", "breakup", "divorce", "cheated", "affair",
    "fired", "lost my job", "can't afford", "in debt",
    "addiction", "overdose", "relapse",
    "hate myself", "worthless", "want to die",
)

# Similarity threshold above which a high-sensitivity fact is allowed through
HIGH_SENSITIVITY_THRESHOLD = 0.72   # query must be very close to the topic
# Standard threshold (already defined below as MEMORY_SIMILARITY_THRESHOLD = 0.48)


def _sensitivity_level(key: str, value: str) -> str:
    """
    Returns 'blocked', 'high', or 'normal'.
    Checks key first (fast), then value substrings (catches generic keys with sensitive values).
    """
    key_lower = key.lower()
    value_lower = value.lower()

    if key_lower in _BLOCKED_KEYS:
        return "blocked"

    if key_lower in _HIGH_SENSITIVITY_KEYS:
        return "high"

    # Value-based detection — catches things like "mood=suicidal" or "feeling=depressed"
    for signal in _SENSITIVE_VALUE_SIGNALS:
        if signal in value_lower:
            return "high"

    return "normal"


def _is_explicit_memory_query(query: str) -> bool:
    """
    Returns True if the user is explicitly asking about their own stored info.
    In that case, high-sensitivity facts are allowed through at a lower threshold.
    """
    triggers = (
        "do you remember", "what do you know", "what did i tell you",
        "i told you", "you know about my", "remember when i said",
        "what do you know about my", "tell me what you know",
    )
    lower = query.lower()
    return any(t in lower for t in triggers)


def _expand_query(query: str) -> str:
    """Expand query with key intent words for better semantic matching."""
    # Strip filler and keep the core intent
    filler = {"hey", "corsbot", "can you", "do you", "what is", "tell me", "i want", "please"}
    words = [w for w in query.lower().split() if w not in filler]
    return " ".join(words[:20])  # cap at 20 words


def _apply_sensitivity_filter(
    scored_facts: list[tuple],
    query: str,
    query_vec,
) -> list[tuple]:
    """
    Filter scored facts through the sensitivity layer before returning to the AI.

    Rules:
    - BLOCKED facts: never included, regardless of query
    - HIGH facts: only included when query similarity > HIGH_SENSITIVITY_THRESHOLD
                  OR the user is explicitly asking about their stored info
    - NORMAL facts: pass through unchanged
    """
    if not scored_facts:
        return scored_facts

    explicit = _is_explicit_memory_query(query)
    filtered = []

    for score, key, value in scored_facts:
        level = _sensitivity_level(key, value)

        if level == "blocked":
            log.debug(f"[memory] blocked sensitive fact: key={key!r}")
            continue

        if level == "high":
            if explicit:
                log.debug(f"[memory] high-sensitivity fact allowed (explicit query): key={key!r}")
                filtered.append((score, key, value))
                continue
            if query_vec is not None:
                fact_vec = _embed_vec(f"{key}: {value}")
                sim = float(np.dot(query_vec, fact_vec))
                if sim >= HIGH_SENSITIVITY_THRESHOLD:
                    log.debug(f"[memory] high-sensitivity fact allowed (sim={sim:.3f}): key={key!r}")
                    filtered.append((score, key, value))
                else:
                    log.debug(f"[memory] high-sensitivity fact suppressed (sim={sim:.3f}): key={key!r}")
            continue

        filtered.append((score, key, value))

    return filtered

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
        query_vec = None
        for key, (value, memory_type, updated_at, reinforcement) in non_identity.items():
            scored_facts.append((_decay_score(memory_type, updated_at, reinforcement), key, value))

    scored_facts.sort(reverse=True)
    scored_facts = _apply_sensitivity_filter(scored_facts, query, query_vec if query else None)
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
        query_vec = None
        for key, (value, memory_type, updated_at, reinforcement) in non_identity.items():
            scored_facts.append((_decay_score(memory_type, updated_at, reinforcement), key, value))

    scored_facts.sort(reverse=True)
    scored_facts = _apply_sensitivity_filter(scored_facts, query, query_vec if query else None)
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

# ---------------- REFLECTION MEMORY ---------------- #

REFLECTION_UPDATE_EVERY = 10   # update reflection every N messages
REFLECTION_MIN_MESSAGES = 5    # don't bother until user has at least this many messages
_reflection_counter: dict = {}

def should_update_reflection(user_id: str) -> bool:
    _reflection_counter.setdefault(user_id, 0)
    _reflection_counter[user_id] += 1
    return _reflection_counter[user_id] % REFLECTION_UPDATE_EVERY == 0


def update_reflection(user_id: str, recent_messages: list[str]):
    """
    Generate and store a behavioral/personality summary for the user
    based on their recent messages. This is the 'reflection' — a high-level
    insight like 'User struggles with confidence but responds well to direct encouragement.'
    """
    if len(recent_messages) < REFLECTION_MIN_MESSAGES:
        return
    try:
        conversation = "\n".join(f"- {m}" for m in recent_messages[-20:])
        output = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "You are analyzing a user's recent messages to a Discord bot. "
                    "Write a single concise behavioral/personality insight about this user — "
                    "something genuinely useful for personalizing future responses. "
                    "Focus on: communication style, emotional patterns, what they respond well to, "
                    "recurring themes, humor style, or how they handle topics. "
                    "Write it as a third-person observation, 1-2 sentences max. "
                    "Examples:\n"
                    "- 'User is self-deprecating but responds well to direct encouragement and humor.'\n"
                    "- 'User tends to vent before asking for advice — acknowledge feelings first.'\n"
                    "- 'User is very competitive and loves banter; match their energy.'\n"
                    "- 'User gets anxious about decisions; short, confident answers work best.'\n"
                    "Only output the insight. No labels, no preamble."
                )},
                {"role": "user", "content": f"Recent messages:\n{conversation}"}
            ],
            max_tokens=80, retries=2, timeout=12,
        )
        insight = output.strip()
        if not insight or len(insight) < 10:
            return

        conn, cursor = get_db()
        now = time.time()
        cursor.execute(
            """INSERT INTO reflections (user_id, insight, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   insight=excluded.insight, updated_at=excluded.updated_at""",
            (str(user_id), insight, now)
        )
        conn.commit()
        log.info(f"[reflection] updated for user={user_id}: {insight[:60]}...")
    except Exception as e:
        log.error(f"[reflection] update error: {type(e).__name__}: {e}")


def get_reflection(user_id: str) -> str:
    """Return the stored behavioral reflection for a user, or empty string."""
    _, cursor = get_db()
    cursor.execute(
        "SELECT insight FROM reflections WHERE user_id=?",
        (str(user_id),)
    )
    row = cursor.fetchone()
    return row[0] if row else ""



