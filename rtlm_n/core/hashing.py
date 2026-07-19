"""Deterministic serialization, hashing, and signing of intelligence states.

States must serialize deterministically (sorted keys, canonical float64
byte order) so that state hashes are stable across nodes and merge orders,
enabling duplicate rejection and lineage tracking. Signing uses HMAC-SHA256
with per-node secrets held by a key registry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict

import numpy as np


def array_bytes(a: np.ndarray) -> bytes:
    """Canonical bytes for an array: shape header + C-ordered float64 data."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    header = json.dumps({"shape": list(a.shape)}, sort_keys=True).encode()
    return header + a.tobytes()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()


def state_hash(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    """Hash of a state's metadata + accumulator arrays (excluding signature)."""
    parts = [canonical_json(meta)]
    for name in sorted(arrays):
        parts.append(name.encode())
        parts.append(array_bytes(arrays[name]))
    return digest(*parts)


class KeyRegistry:
    """Holds per-node signing secrets and verifies signatures.

    In production this would be a PKI; the reference implementation uses
    HMAC-SHA256 so the trust and tamper-detection mechanics are testable.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}

    def register(self, node_id: str, secret: bytes) -> None:
        self._keys[node_id] = secret

    def sign(self, node_id: str, payload_hash: str) -> str:
        if node_id not in self._keys:
            raise KeyError(f"no signing key registered for node {node_id!r}")
        return hmac.new(self._keys[node_id], payload_hash.encode(), hashlib.sha256).hexdigest()

    def verify(self, node_id: str, payload_hash: str, signature: str) -> bool:
        if node_id not in self._keys:
            return False
        expected = self.sign(node_id, payload_hash)
        return hmac.compare_digest(expected, signature)
