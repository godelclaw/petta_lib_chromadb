import os
import json
import re
import uuid

import chromadb


COLLECTION_NAME = os.environ.get("METTACLAW_CHROMA_COLLECTION", "memories")
SYNC_THRESHOLD = int(os.environ.get("METTACLAW_CHROMA_SYNC_THRESHOLD", "20"))
if SYNC_THRESHOLD <= 0:
    raise ValueError("METTACLAW_CHROMA_SYNC_THRESHOLD must be greater than zero")

_collection_metadata = {
    "embedding_model": os.environ.get(
        "METTACLAW_EMBED_MODEL", "unspecified"
    ),
    "embedding_backend": os.environ.get(
        "METTACLAW_EMBED_BACKEND", "sentence-transformers"
    ),
    "embedding_output_dim": int(os.environ.get("METTACLAW_EMBED_DIM", "1024")),
    "embedding_protocol": os.environ.get(
        "METTACLAW_EMBED_PROTOCOL", "qwen3-embedding-8b-1024-v1"
    ),
    "embedding_normalized": os.environ.get(
        "METTACLAW_EMBED_NORMALIZE", "1"
    ).lower()
    not in {"0", "false", "no"},
    "embedding_query_prompt_name": os.environ.get(
        "METTACLAW_EMBED_QUERY_PROMPT_NAME", "query"
    ),
    "hnsw:space": os.environ.get("METTACLAW_CHROMA_METRIC", "cosine"),
    "hnsw:sync_threshold": SYNC_THRESHOLD,
}

CLIENT = chromadb.PersistentClient(
    path=os.environ.get("METTACLAW_CHROMA_DIR")
    or os.environ.get("OMEGACLAW_CHROMA_DIR")
    or "./chroma_db"
)
COLLECTION = CLIENT.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata=_collection_metadata,
    embedding_function=None
)


def _configured_sync_threshold(collection):
    try:
        return int(collection.configuration_json["hnsw"]["sync_threshold"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot read the collection's HNSW sync threshold") from exc


def _ensure_sync_threshold(collection, desired):
    if _configured_sync_threshold(collection) != desired:
        collection.modify(configuration={"hnsw": {"sync_threshold": desired}})

    if _configured_sync_threshold(collection) != desired:
        raise RuntimeError(f"failed to set HNSW sync threshold to {desired}")

    metadata = dict(collection.metadata or {})
    if metadata.get("hnsw:sync_threshold") != desired:
        # hnsw:space is immutable in current Chroma. Omitting it updates the
        # descriptive metadata without asking Chroma to change the metric.
        metadata.pop("hnsw:space", None)
        metadata["hnsw:sync_threshold"] = desired
        collection.modify(metadata=metadata)

    if _configured_sync_threshold(collection) != desired:
        raise RuntimeError("metadata update changed the HNSW sync threshold")


_ensure_sync_threshold(COLLECTION, SYNC_THRESHOLD)


def _memory_log_root():
    return os.environ.get("METTACLAW_MEMORY_LOG_DIR", "./memory/remembered")


def _date_parts(time):
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(time))
    if match:
        return match.groups()
    return ("unknown", "unknown", "unknown")


def _append_memory_log(item_id, content, time):
    year, month, day = _date_parts(time)
    root = _memory_log_root()
    path = os.path.join(root, year, month, f"{day}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "id": item_id,
        "time": str(time),
        "collection": COLLECTION_NAME,
        "content": content,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def remember(content, embedding, time):
    """
    content: str
    embedding: list[float]
    returns: id str
    """
    if not isinstance(content, str):
        raise TypeError("content must be a str")
    if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
        raise TypeError("embedding must be a list of floats")
    item_id = str(uuid.uuid4())
    COLLECTION.add(
        ids=[item_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{"time": time}],
    )
    _append_memory_log(item_id, content, time)
    return item_id

def forget_ids(item_ids):
    """
    item_ids: list[str]
    deletes items by ids
    returns: deleted ids
    """
    if not isinstance(item_ids, list) or not all(isinstance(x, str) for x in item_ids):
        raise TypeError("item_ids must be a list of str")
    if not item_ids:
        return []
    COLLECTION.delete(ids=item_ids)
    return item_ids

def forget_id(item_id):
    """
    item_id: str
    deletes one item by id
    returns: deleted id
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    forget_ids([item_id])
    return item_id

def query(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of [time, content]
    """
    res = query_with_ids(query_embedding, k)
    return [[t, c] for _, t, c in res]

def query_with_ids(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of items
    """
    if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
        raise TypeError("query_embedding must be a list of floats")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be > 0")
    res = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for i in range(len(ids)):
        t = metas[i].get("time") if metas[i] else None
        c = docs[i]
        out.append([ids[i], t, c])
    return out

def query_with_ids_and_dists(query_embedding, k):
    """
    query_embedding: list[float]
    returns: list of items
    """
    if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
        raise TypeError("query_embedding must be a list of floats")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be > 0")
    res = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for i in range(len(ids)):
        t = metas[i].get("time") if metas[i] else None
        c = docs[i]
        out.append([ids[i], t, c, dists[i]])
    return out

def query_by_ids(item_ids):
    """
    item_ids: list[str]
    returns: list of [id, time, content] in requested id order
    """
    if not isinstance(item_ids, list) or not all(isinstance(x, str) for x in item_ids):
        raise TypeError("item_ids must be a list of str")
    if not item_ids:
        return []
    res = COLLECTION.get(
        ids=item_ids,
        include=["documents", "metadatas"],
    )
    found = {}
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    for i in range(len(ids)):
        t = metas[i].get("time") if metas[i] else None
        c = docs[i]
        found[ids[i]] = [ids[i], t, c]
    return [found[item_id] for item_id in item_ids if item_id in found]

def query_by_id(item_id):
    """
    item_id: str
    returns: [id, time, content] or None
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    items = query_by_ids([item_id])
    return items[0] if items else None

def ids_by_time(t):
    """
    t: timestamp
    returns: list[str]
    """
    res = COLLECTION.get(
        where={"time": t},
        include=[],
    )
    return res.get("ids", [])

def link_episode(item_id, linked_time):
    """
    item_id: UUID of the memory being modified
    linked_time: timestamp of the linked memory
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    res = COLLECTION.get(
        ids=[item_id],
        include=["metadatas"],
    )
    if not res.get("ids"):
        raise KeyError(f"memory not found: {item_id}")
    metadata = dict(res["metadatas"][0] or {})
    linked_memories = metadata.get("linkedEpisodes", [])
    if linked_time not in linked_memories:
        linked_memories.append(linked_time)
        metadata["linkedEpisodes"] = linked_memories
        COLLECTION.update(
            ids=[item_id],
            metadatas=[metadata],
        )
    return item_id

def unlink_episode(item_id, linked_time):
    """
    item_id: UUID of the memory being modified
    linked_time: timestamp of the linked memory
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    res = COLLECTION.get(
        ids=[item_id],
        include=["metadatas"],
    )
    if not res.get("ids"):
        raise KeyError(f"memory not found: {item_id}")
    metadata = dict(res["metadatas"][0] or {})
    linked_memories = metadata.get("linkedEpisodes", [])
    if linked_time in linked_memories:
        linked_memories.remove(linked_time)
        if linked_memories:
            metadata["linkedEpisodes"] = linked_memories
        else:
            metadata.pop("linkedEpisodes", None)
        COLLECTION.update(
            ids=[item_id],
            metadatas=[metadata],
        )
    return item_id

def query_linked_episodes(item_id):
    """
    item_id: UUID
    returns: list of linked timestamps
    """
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a str")
    res = COLLECTION.get(
        ids=[item_id],
        include=["metadatas"],
    )
    if not res.get("ids"):
        return []
    metadata = res.get("metadatas", [{}])[0] or {}
    return metadata.get("linkedEpisodes", [])

