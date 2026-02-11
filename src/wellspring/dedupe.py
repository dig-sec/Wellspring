from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from .normalize import canonical_entity_key, normalize_entity_name
from .schemas import Entity
from .storage.base import GraphStore


@dataclass
class EntityResolver:
    store: GraphStore

    def resolve(self, name: str, entity_type: Optional[str] = None) -> Entity:
        normalized_name = normalize_entity_name(name)
        key = canonical_entity_key(normalized_name, entity_type)
        matches = self.store.search_entities(
            query=normalized_name,
            entity_type=entity_type,
            canonical_key=key,
        )
        if matches:
            return matches[0]

        entity = Entity(
            id=str(uuid4()),
            name=normalized_name,
            type=entity_type,
            aliases=[name] if name != normalized_name else [],
            attrs={},
        )
        self.store.upsert_entities([entity])
        return entity
