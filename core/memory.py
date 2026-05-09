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
    """
    Calculate a memory's retrieval score based on decay and reinforcement.
    
    Reinforcement mimics human familiarity — repeated exposure makes memories
    easier to recall. Each reinforcement increases the baseline score and
    slows the decay rate.
    """
    half_life = DECAY.get(memory_type)
    if half_life is None:
        # Identity facts don't decay, but reinforcement still boosts priority
        boost = min(reinforcement, 10) * 0.1  # up to +1.0 bonus
        return min(1.0, 1.0 + boost)
    
    age = time.time() - updated_at
    
    # Reinforcement reduces effective age (memory feels "newer")
    # Each reinforcement level reduces perceived age by 10%, up to 50%
    reinforcement_factor = min(reinforcement, 5) * 0.1
    effective_age = age * (1.0 - reinforcement_factor)
    
    # Base decay
    decay = 0.5 ** (effective_age / half_life)
    
    # Reinforcement boosts the floor (minimum score)
    # Without reinforcement: floor = 0.5
    # With max reinforcement: floor = 0.9
    floor_boost = 0.5 + (min(reinforcement, 5) / 5) * 0.4
    
    return decay * floor_boost


def reinforce_memory(user_id: str, key: str, amount: int = 1) -> None:
    """
    Increase the reinforcement counter for a specific memory.
    
    This mimics human familiarity — when a topic/topic repeatedly appears,
    the memory becomes easier to recall. Call this when:
    - User mentions something that matches a stored memory
    - A memory fact was particularly relevant to the conversation
    - User confirms or re-states a stored fact
    
    Args:
        user_id: The user ID
        key: The memory key to reinforce
        amount: How much to increase reinforcement by (default 1)
    """
    conn, cursor = get_db()
    cursor.execute(
        "UPDATE memory SET reinforcement = reinforcement + ? "
        "WHERE user_id = ? AND key = ?",
        (amount, str(user_id), key)
    )
    conn.commit()
    
    # Also update the embedding's position in FAISS index
    cursor.execute(
        "SELECT value FROM memory WHERE user_id = ? AND key = ?",
        (str(user_id), key)
    )
    row = cursor.fetchone()
    if row:
        vec = _embed_vec(f"{key}: {row[0]}")
        faiss_upsert(str(user_id), key, vec)


def reinforce_memories_for_query(user_id: str, query: str, top_k: int = 3, amount: int = 1) -> list:
    """
    Find and reinforce memories that are relevant to the current query.
    
    This implements "use-dependent strengthening" — memories that are
    retrieved and used in conversation become stronger, mimicking how
    human memory works through retrieval practice.
    
    Args:
        user_id: The user ID
        query: The current message/query
        top_k: Number of top matching memories to reinforce
        amount: Reinforcement amount per memory
    
    Returns:
        List of (key, new_reinforcement) tuples for reinforced memories
    """
    if not query or len(query.split()) < 3:
        return []
    
    _, cursor = get_db()
    
    # Get the query vector
    query_vec = _embed_vec(query)
    
    # Find relevant memories
    matches = faiss_search(str(user_id), query_vec, top_k=top_k)
    
    reinforced = []
    for sim, key in matches:
        if sim < MEMORY_SIMILARITY_THRESHOLD:
            continue
        
        # Reinforce this memory
        cursor.execute(
            "UPDATE memory SET reinforcement = reinforcement + ? "
            "WHERE user_id = ? AND key = ?",
            (amount, str(user_id), key)
        )
        
        # Get new reinforcement value
        cursor.execute(
            "SELECT reinforcement FROM memory WHERE user_id = ? AND key = ?",
            (str(user_id), key)
        )
        row = cursor.fetchone()
        if row:
            reinforced.append((key, row[0]))
            
            # Update FAISS index with new embedding
            cursor.execute(
                "SELECT value FROM memory WHERE user_id = ? AND key = ?",
                (str(user_id), key)
            )
            val_row = cursor.fetchone()
            if val_row:
                vec = _embed_vec(f"{key}: {val_row[0]}")
                faiss_upsert(str(user_id), key, vec)
    
    # Commit all changes
    conn, _ = get_db()
    conn.commit()
    
    if reinforced:
        log.debug(f"[memory] reinforced {len(reinforced)} memories for user={user_id}")
    
    return reinforced

# ---------------- EXTRACT / GET / STORE ---------------- #

MEMORY_EXTRACT_EVERY = 1  # extract on every message — memory is user-scoped across DMs and servers
MEMORY_SIMILARITY_THRESHOLD = 0.48
MAX_MEMORY_FACTS = 6
DEDUP_SIMILARITY_THRESHOLD = 0.92  # facts this similar are considered duplicates

# Keys that are stored but never auto-injected into the prompt.
# nickname/title excluded: bot should never address someone by a stored name unprompted.
# Any key containing another person's name is also excluded at retrieval time.
_PROMPT_EXCLUDED_KEYS = {"nickname", "title", "aura", "bayot", "king", "queen", "god", "goat"}


def _is_about_other_person(key: str, value: str) -> bool:
    """
    Returns True if this fact appears to be about another person rather than the user.
    Keys like 'bennn_aura', 'crisumiles_role', or values referencing other people
    should not be injected into the prompt as facts about the current user.
    """
    # If the key contains an underscore and the prefix looks like a name (capitalized word)
    # it was probably stored as "PersonName_attribute"
    import re as _re
    if _re.match(r'^[a-z]{3,}[_][a-z]', key):
        return True
    return False
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
        # Patterns WITH a capture group — the captured phrase is what was denied
        (r"i'?m not (.+)",           True),
        (r"i am not (.+)",           True),
        (r"i don'?t (.+)",           True),
        (r"i do not (.+)",           True),
        (r"i never (.+)",            True),
        (r"i'?m no longer (.+)",     True),
        (r"i'?m not a (.+)",         True),
        (r"i'?m not the (.+)",       True),
        # Patterns WITHOUT a capture group — scan ALL stored facts for a match
        (r"that'?s not true",        False),
        (r"that is not true",        False),
        (r"that'?s false",           False),
        (r"that is false",           False),
        (r"that'?s wrong",           False),
        (r"that is wrong",           False),
        (r"delete that",             False),
        (r"forget that",             False),
        (r"that'?s not me",          False),
        (r"that is not me",          False),
        (r"not me btw",              False),
        (r"that'?s not my",          False),
        (r"that is not my",          False),
    ]
    import re as _re
    lower = message.lower().strip()

    denied_phrase = None
    scan_all = False

    for pattern, has_capture in denial_patterns:
        match = _re.search(pattern, lower)
        if match:
            if has_capture and match.lastindex:
                denied_phrase = match.group(1).strip()
                # Strip trailing filler words
                denied_phrase = _re.sub(r'\s+(btw|tho|though|lol|fr|ngl|tbh)$', '', denied_phrase).strip()
            else:
                # No specific phrase — scan all stored facts against the recent bot message context
                scan_all = True
            break

    if not denied_phrase and not scan_all:
        return []

    _, cursor = get_db()
    cursor.execute("SELECT key, value FROM memory WHERE user_id=?", (str(user_id),))
    rows = cursor.fetchall()
    if not rows:
        return []

    matches = []

    if denied_phrase:
        # Specific phrase denied — find facts semantically similar to it,
        # OR facts whose value words appear in the denied phrase (keyword overlap)
        denied_vec = _embed_vec(denied_phrase)
        denied_words = set(denied_phrase.lower().split())
        for key, value in rows:
            if key in ("display_name", "username", "server_nickname"):
                continue
            if key.startswith("denied_"):
                continue
            fact_vec = _embed_vec(f"{key}: {value}")
            sim = float(np.dot(denied_vec, fact_vec))
            # Match on semantic similarity OR keyword overlap with the stored value
            value_words = set(value.lower().split())
            keyword_overlap = len(denied_words & value_words) >= 1 and len(value_words) <= 4
            if sim > 0.60 or keyword_overlap:
                matches.append((key, value))
    else:
        # "that's not me" / "that's wrong" — return ALL non-identity facts for user to pick from
        # Limit to the most recently updated facts (most likely what was just referenced)
        cursor.execute(
            "SELECT key, value FROM memory WHERE user_id=? "
            "AND key NOT IN ('display_name','username','server_nickname') "
            "ORDER BY updated_at DESC LIMIT 5",
            (str(user_id),)
        )
        matches = [(k, v) for k, v in cursor.fetchall()
                   if not k.startswith("denied_")]

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
    if len(message.split()) < 2:
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
        result = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Extract personal facts about the USER THEMSELVES from their message. "
                    "Only extract things the user is explicitly stating about themselves. "
                    "Focus on: genuine preferences, hobbies, opinions they hold, things they like/dislike. "
                    f"{skip_hint}"
                    "Reply in key=value format, one per line. "
                    "Use snake_case keys like: likes, dislikes, favorite_game, hobby, opinion_on, etc. "
                    "STRICT RULES — do NOT extract:\n"
                    "- Nicknames, titles, or labels (even if said jokingly — e.g. 'king of X', 'bayot', 'goat')\n"
                    "- Things said about OTHER people\n"
                    "- Jokes, memes, or sarcastic statements\n"
                    "- Slang used in passing (e.g. 'fr', 'no cap', 'bussin')\n"
                    "- Anything said in a roleplay or hypothetical context\n"
                    "Only extract clear, sincere, first-person facts. If none, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=80, retries=2, timeout=10,
        )
        # Handle both tuple return (content, tokens) and direct string return
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        if not isinstance(output, str):
            log.warning(f"[memory] groq_call returned non-string output: {type(output)}")
            return
        
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
        if key in _PROMPT_EXCLUDED_KEYS:
            continue  # stored but never auto-injected
        if _is_about_other_person(key, value):
            continue  # fact is about someone else, not the current user
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


# ── Priority-based memory retrieval ────────────────────────────────────────── #

def get_memory_prioritized(user_id, query: str = "", topic: str = "", emotion: str = "", max_tokens: int = 1500) -> dict:
    """
    Return memory facts organized by priority level.
    
    This function integrates with the priority system to return context
    organized by importance:
    
    - CRITICAL: topic and emotion (passed as parameters)
    - HIGH: highly relevant recent memories (similarity > 0.7)
    - MEDIUM: moderately relevant memories (similarity > 0.5)
    - LOW: older preferences and identity facts
    
    Returns a dict with keys: "critical", "high", "medium", "low"
    """
    from .priority import Priority, ContextPriorityManager, estimate_context_tokens
    
    result = {"critical": "", "high": "", "medium": "", "low": ""}
    
    _, cursor = get_db()
    cursor.execute(
        "SELECT key, value, memory_type, updated_at, reinforcement FROM memory WHERE user_id=?",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    if not rows:
        # Still return topic/emotion as critical
        lines = []
        if topic:
            lines.append(f"topic: {topic}")
        if emotion:
            lines.append(f"emotion: {emotion}")
        if lines:
            result["critical"] = "\n".join(lines)
        return result
    
    now = time.time()
    fact_lookup = {}
    for key, value, memory_type, updated_at, reinforcement in rows:
        resolved_type = _resolve_memory_type(key, memory_type)
        fact_lookup[key] = (value, resolved_type, updated_at, reinforcement or 1)
    
    # Separate identity facts (LOW priority) from other memories
    identity_facts = {}
    non_identity = {}
    
    for key, (value, memory_type, updated_at, reinforcement) in fact_lookup.items():
        if key in _PROMPT_EXCLUDED_KEYS:
            continue
        if _is_about_other_person(key, value):
            continue
        half_life = DECAY.get(memory_type)
        if half_life and (now - updated_at) > half_life * 3:
            continue
        
        if memory_type == "identity" or key in IDENTITY_KEYS:
            identity_facts[key] = (value, updated_at, reinforcement)
        else:
            non_identity[key] = (value, memory_type, updated_at, reinforcement)
    
    # Build priority manager
    manager = ContextPriorityManager(max_tokens=max_tokens)
    
    # Add CRITICAL context: topic and emotion
    if topic:
        manager.add(f"topic: {topic}", Priority.CRITICAL, "topic", now)
    if emotion:
        manager.add(f"emotion: {emotion}", Priority.CRITICAL, "emotion", now)
    
    # Score and categorize non-identity memories
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
            decay = _decay_score(memory_type, updated_at, reinforcement)
            fact_str = f"{key}: {value}"
            
            # Categorize by similarity threshold
            if sim >= 0.7:
                manager.add(fact_str, Priority.HIGH, "memory", updated_at, relevance_score=sim * decay)
            else:
                manager.add(fact_str, Priority.MEDIUM, "memory", updated_at, relevance_score=sim * decay)
    else:
        # No query — add all non-identity as MEDIUM
        for key, (value, memory_type, updated_at, reinforcement) in non_identity.items():
            decay = _decay_score(memory_type, updated_at, reinforcement)
            fact_str = f"{key}: {value}"
            manager.add(fact_str, Priority.MEDIUM, "memory", updated_at, relevance_score=decay)
    
    # Add identity facts as LOW priority
    for key, (value, updated_at, reinforcement) in identity_facts.items():
        fact_str = f"{key}: {value}"
        manager.add(fact_str, Priority.LOW, "identity", updated_at, relevance_score=0.5)
    
    # Get trimmed context
    trimmed = manager.get_trimmed_context(max_tokens)
    
    # Parse the trimmed output back into priority buckets
    if trimmed:
        current_section = None
        section_content = []
        
        for line in trimmed.split("\n"):
            if line.startswith("[CRITICAL]"):
                if current_section and section_content:
                    result[current_section] = "\n".join(section_content)
                current_section = "critical"
                section_content = []
            elif line.startswith("[HIGH]"):
                if current_section and section_content:
                    result[current_section] = "\n".join(section_content)
                current_section = "high"
                section_content = []
            elif line.startswith("[MEDIUM]"):
                if current_section and section_content:
                    result[current_section] = "\n".join(section_content)
                current_section = "medium"
                section_content = []
            elif line.startswith("[LOW]"):
                if current_section and section_content:
                    result[current_section] = "\n".join(section_content)
                current_section = "low"
                section_content = []
            elif line.strip():
                section_content.append(line.strip())
        
        if current_section and section_content:
            result[current_section] = "\n".join(section_content)
    
    return result


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
        if key in _PROMPT_EXCLUDED_KEYS:
            continue  # stored but never auto-injected
        if _is_about_other_person(key, value):
            continue  # fact is about someone else, not the current user
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
        output, _ = groq_call(
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


# ── Relationship Modeling System ───────────────────────────────────────────── #

# Relationship category definitions with keyword signals for classification
_RELATIONSHIP_CATEGORIES = {
    "closest_friends": {
        "relations": {"best friend", "bestie", "bff", "close friend", "ride or die", "day one"},
        "signals": {"trust", "always there", "been through everything", "told them everything", 
                    "closest friend", "bestie", "best friend", "day one friend"},
        "weight": 3,  # High priority for social context
    },
    "favorite_people": {
        "relations": {"crush", "partner", "significant other", "favorite person", "someone i like",
                      "person i like", "interested in", "dating", "in love with"},
        "signals": {"my favorite person", "the one", "crush on", "in love with", "dating",
                    "seeing someone", "talking to someone", "my person"},
        "weight": 3,
    },
    "frequent_conflicts": {
        "relations": {"enemy", "rival", "nemesis", "person i fight with", "someone i hate",
                      "ex friend", "former friend", "toxic"},
        "signals": {"always fighting with", "can't stand", "hate dealing with", "drama with",
                    "conflict with", "issues with", "toxic relationship", "frenemy"},
        "weight": 2,  # Medium priority - relevant but not positive
    },
    "usual_group": {
        "relations": {"friend", "friend group", "crew", "squad", "gang", "circle",
                      "teammate", "classmate", "coworker", "roommate"},
        "signals": {"my friends", "the squad", "the crew", "my group", "we always",
                    "me and the boys", "me and the girls", "our group", "hanging with"},
        "weight": 1,  # Lower priority - general social context
    },
}


def _classify_relationship(relation: str, context: str) -> str:
    """
    Classify a relationship into one of the defined categories.
    Returns the category name or 'acquaintances' if no strong match.
    """
    text = f"{relation} {context}".lower()
    
    scores = {}
    for category, config in _RELATIONSHIP_CATEGORIES.items():
        score = 0
        # Check relation type matches
        for rel_keyword in config["relations"]:
            if rel_keyword in relation.lower():
                score += 2
        # Check context signals
        for signal in config["signals"]:
            if signal in text:
                score += 1
        if score > 0:
            scores[category] = score
    
    if not scores:
        return "acquaintances"
    
    return max(scores, key=scores.get)


def analyze_and_categorize_relationships(user_id: str) -> dict:
    """
    Analyze all stored relationships for a user and categorize them.
    
    Returns a structured dict:
    {
        "closest_friends": [{"name": "...", "relation": "...", "context": "...", "strength": N}],
        "favorite_people": [...],
        "frequent_conflicts": [...],
        "usual_group": [...],
        "acquaintances": [...]
    }
    """
    _, cursor = get_db()
    cursor.execute(
        "SELECT related_name, relation, context, strength, updated_at FROM relationships "
        "WHERE user_id=? ORDER BY strength DESC",
        (str(user_id),)
    )
    rows = cursor.fetchall()
    
    categorized = {cat: [] for cat in list(_RELATIONSHIP_CATEGORIES.keys()) + ["acquaintances"]}
    
    for name, relation, context, strength, updated_at in rows:
        category = _classify_relationship(relation, context or "")
        
        # Apply time decay to strength
        age_days = (time.time() - updated_at) / 86400
        decayed_strength = strength * max(0.5, 1.0 - (age_days / 90))  # 50% decay over 90 days
        
        entry = {
            "name": name,
            "relation": relation,
            "context": context or "",
            "strength": decayed_strength,
            "original_strength": strength,
        }
        
        categorized[category].append(entry)
    
    # Sort each category by strength
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x["strength"], reverse=True)
    
    return categorized


def get_relationship_summary(user_id: str, max_people: int = 10) -> str:
    """
    Get a human-readable relationship summary for prompt injection.
    
    Format:
    Closest friends: Alice (best friend - we've been friends since childhood), Bob (day one)
    Favorite people: Charlie (crush)
    Usual group: The gaming squad - Dave, Eve, Frank
    """
    categorized = analyze_and_categorize_relationships(user_id)
    
    sections = []
    
    # Closest friends (highest priority)
    if categorized["closest_friends"]:
        friends = categorized["closest_friends"][:3]
        friend_strs = []
        for f in friends:
            info = f["name"]
            if f["relation"] and f["relation"] != "friend":
                info += f" ({f['relation']}"
                if f["context"]:
                    info += f" - {f['context']}"
                info += ")"
            elif f["context"]:
                info += f" - {f['context']}"
            friend_strs.append(info)
        sections.append(f"Closest friends: {', '.join(friend_strs)}")
    
    # Favorite people
    if categorized["favorite_people"]:
        favs = categorized["favorite_people"][:2]
        fav_strs = []
        for f in favs:
            info = f["name"]
            if f["relation"]:
                info += f" ({f['relation']}"
                if f["context"]:
                    info += f" - {f['context']}"
                info += ")"
            fav_strs.append(info)
        sections.append(f"Favorite people: {', '.join(fav_strs)}")
    
    # Frequent conflicts (for social awareness)
    if categorized["frequent_conflicts"]:
        conflicts = categorized["frequent_conflicts"][:2]
        conflict_strs = []
        for f in conflicts:
            info = f["name"]
            if f["relation"]:
                info += f" ({f['relation']}"
                if f["context"]:
                    info += f" - {f['context']}"
                info += ")"
            conflict_strs.append(info)
        sections.append(f"Conflicts: {', '.join(conflict_strs)}")
    
    # Usual group
    if categorized["usual_group"]:
        group = categorized["usual_group"][:5]
        names = [g["name"] for g in group]
        group_context = group[0].get("context", "") if group else ""
        group_info = ", ".join(names)
        if group_context:
            group_info += f" ({group_context})"
        sections.append(f"Social circle: {group_info}")
    
    # Acquaintances (just count them)
    if categorized["acquaintances"]:
        count = len(categorized["acquaintances"])
        if count > 0:
            sections.append(f"Other connections: {count} acquaintance{'s' if count != 1 else ''}")
    
    return "\n".join(sections)


def update_relationship_sentiment(user_id: str, person_name: str, sentiment: str, context: str = ""):
    """
    Update or create a relationship with sentiment analysis.
    
    Sentiment can be: "positive", "negative", "neutral", "conflicted"
    This helps track the emotional tone of relationships over time.
    """
    conn, cursor = get_db()
    now = time.time()
    
    # Determine relation type from sentiment
    relation_map = {
        "positive": "close connection",
        "negative": "source of tension",
        "neutral": "acquaintance",
        "conflicted": "complicated relationship",
    }
    relation = relation_map.get(sentiment, "connection")
    
    # Check if relationship exists
    cursor.execute(
        "SELECT strength, context FROM relationships WHERE user_id=? AND related_name=?",
        (str(user_id), person_name)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Update existing relationship
        new_strength = existing[0] + 1
        new_context = existing[1] + "; " + context if context else existing[1]
        cursor.execute(
            "UPDATE relationships SET strength=?, context=?, relation=?, updated_at=? "
            "WHERE user_id=? AND related_name=?",
            (new_strength, new_context, relation, now, str(user_id), person_name)
        )
    else:
        # Create new relationship
        cursor.execute(
            """INSERT INTO relationships (user_id, related_name, relation, context, strength, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(user_id), person_name, relation, context, 1, now)
        )
    
    conn.commit()


def get_social_awareness_context(user_id: str) -> str:
    """
    Generate a comprehensive social awareness context block for the AI prompt.
    
    This provides the bot with social context about the user's relationships,
    enabling more personalized and socially aware responses.
    """
    summary = get_relationship_summary(user_id)
    if not summary:
        return ""
    
    return f"Social context for {user_id}:\n{summary}"


def extract_relationship_categories_from_message(user_id: str, message: str):
    """
    Extract relationship mentions from a message and categorize them.
    Uses AI to understand the nature of relationships mentioned.
    """
    # Quick check for relationship mentions
    lower = message.lower()
    relationship_words = ["friend", "bestie", "bff", "crush", "partner", "boyfriend", "girlfriend",
                          "enemy", "rival", "squad", "crew", "group", "gang", "teammate",
                          "roommate", "classmate", "coworker", "best friend", "day one"]
    
    if not any(word in lower for word in relationship_words):
        return
    
    # Use AI to extract and categorize relationships
    try:
        output, _ = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Extract people mentioned in this message and their relationship to the user. "
                    "For each person, provide: name, relationship_type, sentiment, context. "
                    "Relationship types: closest_friend, favorite_person, frequent_conflict, usual_group_member, acquaintance "
                    "Sentiment: positive, negative, neutral, conflicted "
                    "Format: name|relation_type|sentiment|context (one per line) "
                    "If no clear relationships mentioned, reply: NONE"
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=120, retries=1, timeout=8,
        )
        
        if output.strip().upper() == "NONE":
            return
        
        conn, cursor = get_db()
        now = time.time()
        
        for line in output.strip().split("\n"):
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            
            name = parts[0].strip()
            rel_type = parts[1].strip().lower()
            sentiment = parts[2].strip().lower()
            context = parts[3].strip() if len(parts) > 3 else ""
            
            if not name or not rel_type:
                continue
            
            # Map relation type to a readable relation string
            rel_map = {
                "closest_friend": "close friend",
                "favorite_person": "favorite person",
                "frequent_conflict": "source of conflict",
                "usual_group_member": "group member",
                "acquaintance": "acquaintance",
            }
            relation = rel_map.get(rel_type, rel_type)
            
            # Store/update relationship
            cursor.execute(
                "SELECT strength FROM relationships WHERE user_id=? AND related_name=?",
                (str(user_id), name)
            )
            row = cursor.fetchone()
            strength = (row[0] + 1) if row else 1
            
            cursor.execute(
                """INSERT INTO relationships (user_id, related_name, relation, context, strength, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, related_name) DO UPDATE SET
                       relation=excluded.relation, context=excluded.context,
                       strength=excluded.strength, updated_at=excluded.updated_at""",
                (str(user_id), name, relation, context, strength, now)
            )
        
        conn.commit()
    except Exception as e:
        log.debug(f"[relationships] AI extraction failed: {e}")

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
#
# Two-tier reflection system:
# 1. Light reflection (every 10 messages): Quick style/tone insight
# 2. Deep reflection (every ~100 messages): Comprehensive pattern analysis
#
# Deep reflections analyze behavioral patterns across many interactions,
# generating insights like:
#   "User responds well to teasing but shuts down with serious advice."
#   "Often seeks reassurance indirectly through hypothetical questions."
#   "Avoids discussing stress directly — changes topic when asked."

REFLECTION_LIGHT_EVERY = 10    # quick insight every N messages
REFLECTION_DEEP_EVERY = 100    # deep pattern analysis every N messages
REFLECTION_MIN_MESSAGES = 5    # don't bother until user has at least this many messages
_reflection_counter: dict = {}

# Pattern categories for deep reflection analysis
_REFLECTION_PATTERN_CATEGORIES = {
    "communication_style": {
        "focus": "How does the user communicate? Direct, indirect, verbose, terse?",
        "examples": [
            "User communicates indirectly — hints at needs rather than stating them.",
            "User is very direct and appreciates bluntness in return.",
            "User writes in short bursts; prefers quick back-and-forth over long explanations.",
        ],
    },
    "emotional_patterns": {
        "focus": "What emotional patterns emerge? How do they handle ups and downs?",
        "examples": [
            "User processes emotions through humor — jokes when uncomfortable.",
            "User tends to vent before asking for advice — acknowledge feelings first.",
            "User minimizes their own struggles; deflects concern with jokes.",
        ],
    },
    "response_preferences": {
        "focus": "What type of responses do they engage with most?",
        "examples": [
            "User responds well to teasing but shuts down with serious advice.",
            "User engages most when challenged — enjoys intellectual sparring.",
            "User prefers validation over solutions; just wants to be heard.",
        ],
    },
    "avoidance_patterns": {
        "focus": "What topics or approaches cause disengagement?",
        "examples": [
            "Avoids discussing stress directly — changes topic when asked.",
            "User disengages when conversations get too serious too fast.",
            "User avoids vulnerability — deflects personal questions with humor.",
        ],
    },
    "engagement_triggers": {
        "focus": "What consistently gets them talking more?",
        "examples": [
            "User lights up when talking about creative projects — ask follow-ups.",
            "User engages deeply with philosophical questions; surface-level chat bores them.",
            "User opens up when feeling heard — reflective listening works well.",
        ],
    },
}


def should_update_reflection(user_id: str) -> bool:
    """Check if we should do a light reflection update."""
    _reflection_counter.setdefault(user_id, 0)
    _reflection_counter[user_id] += 1
    return _reflection_counter[user_id] % REFLECTION_LIGHT_EVERY == 0


def should_do_deep_reflection(user_id: str) -> bool:
    """Check if we should do a deep pattern analysis."""
    _reflection_counter.setdefault(user_id, 0)
    return _reflection_counter[user_id] % REFLECTION_DEEP_EVERY == 0 and _reflection_counter[user_id] >= REFLECTION_DEEP_EVERY


def update_reflection(user_id: str, recent_messages: list[str]):
    """
    Generate and store a behavioral/personality summary for the user
    based on their recent messages. This is the 'reflection' — a high-level
    insight like 'User struggles with confidence but responds well to direct encouragement.'
    
    This is the LIGHT reflection — quick, focused on immediate patterns.
    """
    if len(recent_messages) < REFLECTION_MIN_MESSAGES:
        return
    
    is_deep = should_do_deep_reflection(user_id)
    
    try:
        conversation = "\n".join(f"- {m}" for m in recent_messages[-30:])
        
        if is_deep:
            # Deep reflection — comprehensive pattern analysis
            system_prompt = (
                "You are analyzing a user's conversation patterns with a Discord bot. "
                "Generate 2-3 concise behavioral insights that would help the bot respond better. "
                "Focus on PATTERNS across messages, not one-off observations. "
                "Look for:\n"
                "- Communication style (direct/indirect, verbose/terse)\n"
                "- Emotional patterns (how they process feelings)\n"
                "- Response preferences (what type of replies they engage with)\n"
                "- Avoidance patterns (what makes them disengage)\n"
                "- Engagement triggers (what gets them talking more)\n\n"
                "Format: One insight per line. 1-2 sentences each. Third-person. "
                "Be specific and actionable — not generic platitudes.\n\n"
                "Good examples:\n"
                "- 'User responds well to teasing but shuts down with serious advice.'\n"
                "- 'Often seeks reassurance indirectly through hypothetical questions.'\n"
                "- 'Avoids discussing stress directly — changes topic when asked.'\n"
                "- 'User processes emotions through humor — jokes when uncomfortable.'\n"
                "- 'Engages deeply with philosophical takes; surface-level chat bores them.'\n"
                "Only output the insights. No labels, no preamble, no explanations."
            )
            max_tokens = 150
        else:
            # Light reflection — quick style insight
            system_prompt = (
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
            )
            max_tokens = 80
        
        output, _ = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Recent messages:\n{conversation}"}
            ],
            max_tokens=max_tokens, retries=2, timeout=12,
        )
        insight = output.strip()
        if not insight or len(insight) < 10:
            return
        
        # For deep reflections, take the first line as the primary insight
        # but store the full analysis for potential future use
        if is_deep:
            lines = [l.strip() for l in insight.split("\n") if l.strip()]
            primary_insight = lines[0] if lines else insight
            log.info(f"[reflection] DEEP analysis for user={user_id}: {len(lines)} insights generated")
        else:
            primary_insight = insight
        
        conn, cursor = get_db()
        now = time.time()
        cursor.execute(
            """INSERT INTO reflections (user_id, insight, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   insight=excluded.insight, updated_at=excluded.updated_at""",
            (str(user_id), primary_insight, now)
        )
        conn.commit()
        log.info(f"[reflection] updated for user={user_id}: {primary_insight[:60]}...")
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


def generate_reflection_summary(user_id: str) -> str:
    """
    Generate a comprehensive reflection summary by analyzing patterns
    across the user's stored memories, relationships, and conversation history.
    
    This is called during deep reflection to synthesize all available data.
    """
    _, cursor = get_db()
    
    # Gather all available data
    cursor.execute("SELECT key, value FROM memory WHERE user_id=?", (str(user_id),))
    memories = cursor.fetchall()
    
    cursor.execute("SELECT related_name, relation, context, strength FROM relationships WHERE user_id=?", (str(user_id),))
    relationships = cursor.fetchall()
    
    cursor.execute("SELECT insight FROM reflections WHERE user_id=?", (str(user_id),))
    prev_reflection = cursor.fetchone()
    
    # Build context for analysis
    context_parts = []
    
    if memories:
        memory_text = "\n".join(f"- {k}: {v}" for k, v in memories[:20])
        context_parts.append(f"Stored facts about user:\n{memory_text}")
    
    if relationships:
        rel_text = "\n".join(f"- {name} ({relation}): {context}" for name, relation, context, _ in relationships[:10])
        context_parts.append(f"User's relationships:\n{rel_text}")
    
    if prev_reflection and prev_reflection[0]:
        context_parts.append(f"Previous insight: {prev_reflection[0]}")
    
    if not context_parts:
        return ""
    
    context = "\n\n".join(context_parts)
    
    try:
        output, _ = groq_call(
            "llama-3.1-8b-instant",
            [
                {"role": "system", "content": (
                    "Analyze this user's data and generate 2-3 actionable behavioral insights. "
                    "Look for patterns in their stored facts, relationships, and previous reflections. "
                    "Focus on what would help a Discord bot interact with them better.\n\n"
                    "Consider:\n"
                    "- What communication style works best?\n"
                    "- What topics engage them vs. bore them?\n"
                    "- How do they seek support or connection?\n"
                    "- What patterns emerge across their data?\n\n"
                    "Format: One insight per line. Specific and actionable. Third-person."
                )},
                {"role": "user", "content": context}
            ],
            max_tokens=150, retries=1, timeout=10,
        )
        
        insights = [l.strip() for l in output.strip().split("\n") if l.strip()]
        return "\n".join(insights[:3])
    except Exception as e:
        log.debug(f"[reflection] summary generation failed: {e}")
        return ""



