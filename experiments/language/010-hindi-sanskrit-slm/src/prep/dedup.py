"""Exact-match deduplication and benchmark leakage checks."""

import hashlib


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedupe(rows, key):
    """Keep the first row for each key. Returns (kept, dropped_count)."""
    seen = set()
    kept = []
    dropped = 0
    for row in rows:
        digest = content_hash(key(row))
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(row)
    return kept, dropped


def leakage(rows, banned_hashes, key):
    """Drop rows whose key hash is in the banned set. Returns (kept, dropped)."""
    kept = []
    dropped = 0
    for row in rows:
        if content_hash(key(row)) in banned_hashes:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def hash_set(texts):
    return {content_hash(text) for text in texts if text}
