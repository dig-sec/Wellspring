"""Sync from OpenCTI into Wellspring."""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from .client import OpenCTIClient
from ..dedupe import EntityResolver
from ..normalize import normalize_predicate
from ..schemas import Entity, ExtractionRun, Relation, Subgraph
from ..storage.base import GraphStore

logger = logging.getLogger(__name__)

# OpenCTI entity_type → Wellspring entity type
_TYPE_MAP: Dict[str, str] = {
    "Malware": "malware",
    "Threat-Actor": "threat_actor",
    "Threat-Actor-Group": "threat_actor",
    "Threat-Actor-Individual": "threat_actor",
    "Attack-Pattern": "attack_pattern",
    "Tool": "tool",
    "Vulnerability": "vulnerability",
    "Campaign": "campaign",
    "Intrusion-Set": "threat_actor",
    "Indicator": "indicator",
    "Infrastructure": "infrastructure",
    "Course-Of-Action": "mitigation",
    "Report": "report",
}


@dataclass
class SyncResult:
    """Summary of a sync operation."""
    entities_pulled: int = 0
    relations_pulled: int = 0
    reports_queued: int = 0
    errors: List[str] = field(default_factory=list)


def pull_from_opencti(
    opencti: OpenCTIClient,
    graph_store: GraphStore,
    entity_types: List[str],
    max_per_type: int = 0,
    run_store: Any = None,
    settings: Any = None,
    progress_cb: Any = None,
) -> SyncResult:
    """Pull recent entities from OpenCTI and import into Wellspring.

    This is a synchronous function designed to run in a background thread.
    Uses direct GraphQL queries — no STIX conversion needed.
    """
    result = SyncResult()
    resolver = EntityResolver(graph_store)
    fetch_limit = max_per_type if max_per_type > 0 else 10000  # 0 = unlimited

    def _progress(msg: str):
        if progress_cb:
            progress_cb(msg)

    for i, entity_type in enumerate(entity_types, 1):
        _progress(f"[{i}/{len(entity_types)}] Fetching {entity_type}...")
        try:
            # ── Reports: special handling for contained objects ──
            if entity_type == "Report":
                reports = opencti.list_reports_with_objects(first=fetch_limit)
                logger.info("Got %d reports from OpenCTI", len(reports))
                _progress(f"[{i}/{len(entity_types)}] Processing {len(reports)} reports...")

                for ri, rpt in enumerate(reports, 1):
                    if ri % 10 == 0:
                        _progress(f"[{i}/{len(entity_types)}] Report {ri}/{len(reports)}: {result.entities_pulled} entities so far")
                    # Create the report entity
                    report_entity = resolver.resolve(rpt["name"], entity_type="report")
                    if rpt.get("description"):
                        report_entity.attrs["description"] = rpt["description"]
                    report_entity.attrs["opencti_id"] = rpt["id"]
                    report_entity.attrs["opencti_type"] = "Report"
                    if rpt.get("published"):
                        report_entity.attrs["published"] = rpt["published"]
                    graph_store.upsert_entities([report_entity])
                    result.entities_pulled += 1

                    # Create contained object entities + "mentions" edges
                    for obj in rpt.get("objects", []):
                        ws_type = _TYPE_MAP.get(obj["type"], obj["type"].lower())
                        obj_entity = resolver.resolve(obj["name"], entity_type=ws_type)
                        obj_entity.attrs["opencti_id"] = obj["id"]
                        graph_store.upsert_entities([obj_entity])
                        result.entities_pulled += 1

                        rel = Relation(
                            id=str(uuid4()),
                            subject_id=report_entity.id,
                            predicate="mentions",
                            object_id=obj_entity.id,
                            confidence=0.9,
                            attrs={"origin": "opencti"},
                        )
                        graph_store.upsert_relations([rel])
                        result.relations_pulled += 1

                    # Import explicit relationships within the report
                    for rel_data in rpt.get("relations", []):
                        from_name = rel_data.get("from_name", "")
                        to_name = rel_data.get("to_name", "")
                        if not from_name or not to_name:
                            continue
                        from_ws = _TYPE_MAP.get(rel_data.get("from_type", ""), None)
                        to_ws = _TYPE_MAP.get(rel_data.get("to_type", ""), None)
                        subj = resolver.resolve(from_name, entity_type=from_ws)
                        obj_ent = resolver.resolve(to_name, entity_type=to_ws)
                        predicate = normalize_predicate(rel_data["type"])
                        if not predicate:
                            predicate = "related_to"
                        confidence = (rel_data.get("confidence") or 50) / 100.0
                        rel = Relation(
                            id=str(uuid4()),
                            subject_id=subj.id,
                            predicate=predicate,
                            object_id=obj_ent.id,
                            confidence=min(max(confidence, 0.0), 1.0),
                            attrs={"origin": "opencti", "opencti_rel_id": rel_data.get("id", "")},
                        )
                        graph_store.upsert_relations([rel])
                        result.relations_pulled += 1

                    # Queue report text for LLM extraction
                    report_text = rpt.get("text", "").strip()
                    if report_text and run_store and settings and len(report_text) > 50:
                        run_id = str(uuid4())
                        run = ExtractionRun(
                            run_id=run_id,
                            started_at=datetime.utcnow(),
                            model=settings.ollama_model,
                            prompt_version=settings.prompt_version,
                            params={
                                "chunk_size": settings.chunk_size,
                                "chunk_overlap": settings.chunk_overlap,
                            },
                            status="pending",
                            error=None,
                        )
                        source_uri = f"opencti://report/{rpt['id']}"
                        run_store.create_run(
                            run, source_uri, report_text,
                            {"opencti_report": rpt["name"], "opencti_id": rpt["id"]},
                        )
                        result.reports_queued += 1
                        logger.info(
                            "Queued report '%s' (%d chars) for LLM extraction",
                            rpt["name"], len(report_text),
                        )

                continue

            # ── Standard entity types ──
            entities = opencti.list_entities_with_relations(
                entity_type, first=fetch_limit,
            )
            logger.info("Got %d %s from OpenCTI", len(entities), entity_type)
            _progress(f"[{i}/{len(entity_types)}] Processing {len(entities)} {entity_type}...")

            for ei, ent in enumerate(entities, 1):
                if ei % 50 == 0:
                    _progress(f"[{i}/{len(entity_types)}] {entity_type} {ei}/{len(entities)}: {result.entities_pulled} entities so far")
                ws_type = _TYPE_MAP.get(entity_type, entity_type.lower())
                entity = resolver.resolve(ent["name"], entity_type=ws_type)

                # Store OpenCTI metadata
                if ent.get("description"):
                    entity.attrs["description"] = ent["description"]
                entity.attrs["opencti_id"] = ent["id"]
                entity.attrs["opencti_type"] = entity_type
                graph_store.upsert_entities([entity])
                result.entities_pulled += 1

                # Process relationships
                for rel in ent.get("relations", []):
                    from_name = rel.get("from_name", "")
                    to_name = rel.get("to_name", "")
                    rel_type = rel.get("type", "related-to")

                    if not from_name or not to_name:
                        continue

                    from_ws_type = _TYPE_MAP.get(
                        rel.get("from_type", ""), None
                    )
                    to_ws_type = _TYPE_MAP.get(
                        rel.get("to_type", ""), None
                    )

                    subj = resolver.resolve(from_name, entity_type=from_ws_type)
                    obj = resolver.resolve(to_name, entity_type=to_ws_type)

                    predicate = normalize_predicate(rel_type)
                    if not predicate:
                        predicate = "related_to"

                    confidence = (rel.get("confidence") or 50) / 100.0

                    relation = Relation(
                        id=str(uuid4()),
                        subject_id=subj.id,
                        predicate=predicate,
                        object_id=obj.id,
                        confidence=min(max(confidence, 0.0), 1.0),
                        attrs={
                            "origin": "opencti",
                            "opencti_rel_id": rel.get("id", ""),
                        },
                    )
                    graph_store.upsert_relations([relation])
                    result.relations_pulled += 1

        except Exception as exc:
            logger.warning("Failed to pull %s: %s", entity_type, exc)
            result.errors.append(f"{entity_type}: {exc}")
            continue

    logger.info(
        "Pulled %d entities, %d relations, queued %d reports for LLM (%d errors)",
        result.entities_pulled,
        result.relations_pulled,
        result.reports_queued,
        len(result.errors),
    )
    return result
