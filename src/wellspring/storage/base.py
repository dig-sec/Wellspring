from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..schemas import Entity, Relation, Provenance, ExtractionRun, Subgraph


class GraphStore(ABC):
    @abstractmethod
    def upsert_entities(self, entities: List[Entity]) -> List[Entity]:
        raise NotImplementedError

    @abstractmethod
    def upsert_relations(self, relations: List[Relation]) -> List[Relation]:
        raise NotImplementedError

    @abstractmethod
    def attach_provenance(self, relation_id: str, provenance: Provenance) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        raise NotImplementedError

    @abstractmethod
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        canonical_key: Optional[str] = None,
    ) -> List[Entity]:
        raise NotImplementedError

    @abstractmethod
    def get_subgraph(
        self,
        seed_entity_id: str,
        depth: int = 1,
        min_confidence: float = 0.0,
        source_uri: Optional[str] = None,
    ) -> Subgraph:
        raise NotImplementedError

    @abstractmethod
    def explain_edge(
        self, relation_id: str
    ) -> Tuple[Relation, List[Provenance], List[ExtractionRun]]:
        raise NotImplementedError
