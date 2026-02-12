from __future__ import annotations

from collections import Counter
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response

from ..config import get_settings
from ..schemas import (
    ExplainEntityRelation,
    ExplainEntityResponse,
    ExplainResponse,
    ExtractionRun,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RunStatusResponse,
    Subgraph,
)
from ..stix.importer import ingest_stix_bundle, parse_stix_file
from ..stix.exporter import export_stix_bundle
from ..export import export_csv_zip, export_graphml, export_json, export_markdown
from ..opencti.client import OpenCTIClient
from ..opencti.sync import pull_from_opencti
from ..storage.factory import create_graph_store, create_metrics_store, create_run_store
from .visualize import render_html
from .ui import render_root_ui
from .tasks import task_manager, TaskStatus

settings = get_settings()

graph_store = create_graph_store(settings)
run_store = create_run_store(settings)
metrics_store = create_metrics_store(settings)

router = APIRouter()


_TIMELINE_INTERVALS = {"day", "week", "month", "quarter", "year"}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_start(dt: datetime, interval: str) -> datetime:
    dt = _to_utc(dt)
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval == "week":
        base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return base - timedelta(days=base.weekday())
    if interval == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if interval == "quarter":
        first_month = ((dt.month - 1) // 3) * 3 + 1
        return dt.replace(month=first_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _resolve_entity(seed_id: Optional[str], seed_name: Optional[str]):
    if seed_id:
        entity = graph_store.get_entity(seed_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        return entity
    if seed_name:
        matches = graph_store.search_entities(seed_name)
        if not matches:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        return matches[0]
    raise HTTPException(status_code=400, detail="seed_id or seed_name required")


def _get_opencti_client() -> Optional[OpenCTIClient]:
    """Get an OpenCTI client if configured."""
    if not settings.opencti_url or not settings.opencti_token:
        return None
    return OpenCTIClient(settings.opencti_url, settings.opencti_token)


@router.get("/", response_class=HTMLResponse)
def root() -> str:
    return render_root_ui()


@router.get("/api/search")
def search_entities(q: str = Query(..., min_length=1)):
    """Search for entities by name."""
    matches = graph_store.search_entities(q)
    return [{"id": e.id, "name": e.name, "type": e.type} for e in matches[:50]]


def _extract_text(raw: bytes, filename: str) -> str:
    """Extract plain text from raw file bytes, with PDF support."""
    if filename.lower().endswith(".pdf"):
        import fitz  # pymupdf

        doc = fitz.open(stream=raw, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    return raw.decode("utf-8", errors="replace")


def _is_stix_bundle(raw: bytes) -> bool:
    """Quick check if raw bytes look like a STIX 2.1 JSON bundle."""
    try:
        # Only peek at the first 200 bytes to avoid parsing huge files
        head = raw[:200].decode("utf-8", errors="replace")
        return '"type"' in head and '"bundle"' in head
    except Exception:
        return False


@router.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents (text, PDF, or STIX 2.1 JSON) for ingestion."""
    results = []
    for f in files:
        raw = await f.read()
        filename = f.filename or ""

        # ── STIX bundle fast-path: structured import, no LLM needed ──
        if filename.lower().endswith(".json") and _is_stix_bundle(raw):
            try:
                bundle = parse_stix_file(raw, filename)
                stix_result = ingest_stix_bundle(
                    bundle, graph_store, source_uri=f"stix://{filename}"
                )
                results.append({
                    "filename": filename,
                    "status": "completed",
                    "type": "stix",
                    "entities": stix_result.entities_created,
                    "relations": stix_result.relations_created,
                    "skipped": stix_result.objects_skipped,
                    "errors": stix_result.errors,
                })
            except ValueError as exc:
                results.append({
                    "filename": filename,
                    "status": "error",
                    "type": "stix",
                    "error": str(exc),
                })
            continue

        # ── Regular document: enqueue for LLM extraction ──
        text = _extract_text(raw, filename)
        source_uri = f"upload://{filename}"

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
        run_store.create_run(
            run, source_uri, text, {"filename": filename, "size": len(raw)}
        )
        results.append({"run_id": run_id, "filename": filename, "status": "pending"})
    return results


@router.get("/api/runs")
def list_runs():
    """List recent extraction runs."""
    runs = run_store.list_recent_runs(limit=50)
    return [{"run_id": r.run_id, "status": r.status, "model": r.model,
             "started_at": r.started_at.isoformat()} for r in runs]


@router.get("/api/export/stix")
def export_stix(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Export a subgraph as a STIX 2.1 JSON bundle."""
    if seed_id:
        seed = seed_id
        if not graph_store.get_entity(seed):
            raise HTTPException(status_code=404, detail="Seed entity not found")
    elif seed_name:
        matches = graph_store.search_entities(seed_name)
        if not matches:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        seed = matches[0].id
    else:
        raise HTTPException(status_code=400, detail="seed_id or seed_name required")

    subgraph = graph_store.get_subgraph(
        seed_entity_id=seed,
        depth=depth,
        min_confidence=min_confidence,
    )
    bundle = export_stix_bundle(subgraph)
    return bundle


def _resolve_subgraph(
    seed_id: Optional[str],
    seed_name: Optional[str],
    depth: int,
    min_confidence: float,
    scope: str,
):
    """Resolve a subgraph — either seeded or full DB."""
    if scope == "all":
        return graph_store.get_full_graph(min_confidence=min_confidence)
    if seed_id:
        seed = seed_id
        if not graph_store.get_entity(seed):
            raise HTTPException(status_code=404, detail="Seed entity not found")
    elif seed_name:
        matches = graph_store.search_entities(seed_name)
        if not matches:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        seed = matches[0].id
    else:
        raise HTTPException(status_code=400,
                            detail="seed_id / seed_name required, or set scope=all")
    return graph_store.get_subgraph(
        seed_entity_id=seed, depth=depth, min_confidence=min_confidence,
    )


@router.get("/api/export/csv")
def export_csv(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    scope: str = Query(default="seed", regex="^(seed|all)$"),
):
    """Export entities & relations as CSV files in a ZIP archive."""
    subgraph = _resolve_subgraph(seed_id, seed_name, depth, min_confidence, scope)
    data = export_csv_zip(subgraph)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.zip"},
    )


@router.get("/api/export/graphml")
def export_graphml_endpoint(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    scope: str = Query(default="seed", regex="^(seed|all)$"),
):
    """Export as GraphML (Gephi / Cytoscape / yEd)."""
    subgraph = _resolve_subgraph(seed_id, seed_name, depth, min_confidence, scope)
    xml = export_graphml(subgraph)
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.graphml"},
    )


@router.get("/api/export/json")
def export_json_endpoint(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    scope: str = Query(default="seed", regex="^(seed|all)$"),
):
    """Export as a plain JSON knowledge graph."""
    subgraph = _resolve_subgraph(seed_id, seed_name, depth, min_confidence, scope)
    return export_json(subgraph)


@router.get("/api/export/markdown")
def export_markdown_endpoint(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    scope: str = Query(default="seed", regex="^(seed|all)$"),
):
    """Export as a human-readable Markdown report."""
    subgraph = _resolve_subgraph(seed_id, seed_name, depth, min_confidence, scope)
    md = export_markdown(subgraph)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.md"},
    )


# ── POST export: accepts the visible graph inline ───────────────────

@router.post("/api/export/stix")
def post_export_stix(payload: Subgraph):
    """Export a client-provided subgraph as STIX 2.1."""
    bundle = export_stix_bundle(payload)
    return bundle


@router.post("/api/export/csv")
def post_export_csv(payload: Subgraph):
    """Export a client-provided subgraph as CSV ZIP."""
    data = export_csv_zip(payload)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.zip"},
    )


@router.post("/api/export/graphml")
def post_export_graphml(payload: Subgraph):
    """Export a client-provided subgraph as GraphML."""
    xml = export_graphml(payload)
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.graphml"},
    )


@router.post("/api/export/json")
def post_export_json(payload: Subgraph):
    """Export a client-provided subgraph as plain JSON."""
    return export_json(payload)


@router.post("/api/export/markdown")
def post_export_markdown(payload: Subgraph):
    """Export a client-provided subgraph as Markdown."""
    md = export_markdown(payload)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=wellspring-export.md"},
    )


@router.delete("/api/runs")
def delete_all_runs():
    """Delete all runs and associated documents/chunks."""
    count = run_store.delete_all_runs()
    return {"deleted": count}


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
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
    run_store.create_run(run, payload.source_uri, payload.text, payload.metadata)
    return IngestResponse(run_id=run_id, status=run.status)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def run_status(run_id: str) -> RunStatusResponse:
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(run=run)


@router.post("/query", response_model=Subgraph)
def query(payload: QueryRequest) -> Subgraph:
    if payload.seed_id:
        seed = payload.seed_id
        if not graph_store.get_entity(seed):
            raise HTTPException(status_code=404, detail="Seed entity not found")
    elif payload.seed_name:
        matches = graph_store.search_entities(payload.seed_name)
        if not matches:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        seed = matches[0].id
    else:
        raise HTTPException(status_code=400, detail="seed_id or seed_name required")

    return graph_store.get_subgraph(
        seed_entity_id=seed,
        depth=payload.depth,
        min_confidence=payload.min_confidence,
        source_uri=payload.source_uri,
        since=payload.since,
        until=payload.until,
    )


@router.get("/explain", response_model=ExplainResponse | ExplainEntityResponse)
def explain(
    relation_id: Optional[str] = Query(default=None),
    entity_id: Optional[str] = Query(default=None),
):
    if relation_id:
        try:
            relation, provenance, runs = graph_store.explain_edge(relation_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Relation not found") from None
        return ExplainResponse(relation=relation, provenance=provenance, runs=runs)
    if entity_id:
        entity = graph_store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        subgraph = graph_store.get_subgraph(entity_id, depth=1)
        relations = []
        for edge in subgraph.edges[:50]:
            relation, provenance, runs = graph_store.explain_edge(edge.id)
            relations.append(
                ExplainEntityRelation(relation=relation, provenance=provenance, runs=runs)
            )
        return ExplainEntityResponse(entity=entity, relations=relations)
    raise HTTPException(status_code=400, detail="relation_id or entity_id required")


@router.get("/visualize", response_class=HTMLResponse)
def visualize(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=1, ge=0, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    source_uri: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
):
    if seed_id:
        seed = seed_id
        if not graph_store.get_entity(seed):
            raise HTTPException(status_code=404, detail="Seed entity not found")
    elif seed_name:
        matches = graph_store.search_entities(seed_name)
        if not matches:
            raise HTTPException(status_code=404, detail="Seed entity not found")
        seed = matches[0].id
    else:
        raise HTTPException(status_code=400, detail="seed_id or seed_name required")

    subgraph = graph_store.get_subgraph(
        seed_entity_id=seed,
        depth=depth,
        min_confidence=min_confidence,
        source_uri=source_uri,
        since=since,
        until=until,
    )
    title = f"Wellspring Graph: {seed_name or seed_id}"
    return render_html(subgraph, title=title)


@router.get("/api/timeline/entity")
def entity_timeline(
    entity_id: Optional[str] = Query(default=None),
    entity_name: Optional[str] = Query(default=None),
    depth: int = Query(default=1, ge=0, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    source_uri: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    interval: str = Query(default="month", regex="^(day|week|month|quarter|year)$"),
):
    """Show how an entity's connected relations evolve over time."""
    interval = interval.lower()
    if interval not in _TIMELINE_INTERVALS:
        raise HTTPException(status_code=400, detail="interval must be day|week|month|quarter|year")

    entity = _resolve_entity(entity_id, entity_name)
    since_utc = _to_utc(since) if since else None
    until_utc = _to_utc(until) if until else None
    if since_utc and until_utc and since_utc > until_utc:
        raise HTTPException(status_code=400, detail="since must be <= until")

    subgraph = graph_store.get_subgraph(
        seed_entity_id=entity.id,
        depth=depth,
        min_confidence=min_confidence,
        source_uri=source_uri,
        since=since,
        until=until,
    )

    buckets: Dict[datetime, Dict[str, object]] = {}

    for edge in subgraph.edges:
        try:
            _, provenance, _ = graph_store.explain_edge(edge.id)
        except KeyError:
            continue

        for prov in provenance:
            if source_uri and prov.source_uri != source_uri:
                continue
            ts = _to_utc(prov.timestamp)
            if since_utc and ts < since_utc:
                continue
            if until_utc and ts > until_utc:
                continue

            bucket_key = _bucket_start(ts, interval)
            if bucket_key not in buckets:
                buckets[bucket_key] = {
                    "relation_ids": set(),
                    "incoming_ids": set(),
                    "outgoing_ids": set(),
                    "evidence_count": 0,
                    "predicates": Counter(),
                }
            bucket = buckets[bucket_key]
            relation_ids = bucket["relation_ids"]
            incoming_ids = bucket["incoming_ids"]
            outgoing_ids = bucket["outgoing_ids"]
            predicates = bucket["predicates"]

            relation_ids.add(edge.id)
            if edge.subject_id == entity.id:
                outgoing_ids.add(edge.id)
            if edge.object_id == entity.id:
                incoming_ids.add(edge.id)
            bucket["evidence_count"] = int(bucket["evidence_count"]) + 1
            predicates[edge.predicate] += 1

    timeline = []
    for bucket_start in sorted(buckets):
        data = buckets[bucket_start]
        predicate_counts = data["predicates"]
        top_predicates = [
            {"predicate": pred, "count": count}
            for pred, count in predicate_counts.most_common(10)
        ]
        timeline.append(
            {
                "bucket_start": bucket_start.isoformat(),
                "relation_count": len(data["relation_ids"]),
                "incoming_relation_count": len(data["incoming_ids"]),
                "outgoing_relation_count": len(data["outgoing_ids"]),
                "evidence_count": int(data["evidence_count"]),
                "top_predicates": top_predicates,
            }
        )

    return {
        "entity": {"id": entity.id, "name": entity.name, "type": entity.type},
        "interval": interval,
        "depth": depth,
        "min_confidence": min_confidence,
        "source_uri": source_uri,
        "since": since_utc.isoformat() if since_utc else None,
        "until": until_utc.isoformat() if until_utc else None,
        "bucket_count": len(timeline),
        "buckets": timeline,
    }


@router.get("/api/timeline/threat-actors")
def threat_actor_timeline(
    seed_id: Optional[str] = Query(default=None),
    seed_name: Optional[str] = Query(default=None),
    depth: int = Query(default=2, ge=0, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    source_uri: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    interval: str = Query(default="month", regex="^(day|week|month|quarter|year)$"),
    top_n: int = Query(default=10, ge=1, le=100),
):
    """Aggregate temporal activity for threat actors in a graph scope."""
    interval = interval.lower()
    if interval not in _TIMELINE_INTERVALS:
        raise HTTPException(status_code=400, detail="interval must be day|week|month|quarter|year")

    since_utc = _to_utc(since) if since else None
    until_utc = _to_utc(until) if until else None
    if since_utc and until_utc and since_utc > until_utc:
        raise HTTPException(status_code=400, detail="since must be <= until")

    seed_entity = None
    if seed_id or seed_name:
        seed_entity = _resolve_entity(seed_id, seed_name)
        subgraph = graph_store.get_subgraph(
            seed_entity_id=seed_entity.id,
            depth=depth,
            min_confidence=min_confidence,
            source_uri=source_uri,
            since=since,
            until=until,
        )
    else:
        subgraph = graph_store.get_full_graph(min_confidence=min_confidence)

    node_by_id = {node.id: node for node in subgraph.nodes}
    threat_actor_ids = {node.id for node in subgraph.nodes if node.type == "threat_actor"}

    actor_buckets: Dict[str, Dict[datetime, Dict[str, object]]] = {}
    actor_relation_ids: Dict[str, set[str]] = {}
    actor_evidence_totals: Dict[str, int] = {}

    for edge in subgraph.edges:
        connected_actor_ids: List[str] = []
        if edge.subject_id in threat_actor_ids:
            connected_actor_ids.append(edge.subject_id)
        if edge.object_id in threat_actor_ids and edge.object_id != edge.subject_id:
            connected_actor_ids.append(edge.object_id)
        if not connected_actor_ids:
            continue

        try:
            _, provenance, _ = graph_store.explain_edge(edge.id)
        except KeyError:
            continue

        filtered_timestamps: List[datetime] = []
        for prov in provenance:
            if source_uri and prov.source_uri != source_uri:
                continue
            ts = _to_utc(prov.timestamp)
            if since_utc and ts < since_utc:
                continue
            if until_utc and ts > until_utc:
                continue
            filtered_timestamps.append(ts)

        if not filtered_timestamps:
            continue

        for actor_id in connected_actor_ids:
            actor_relation_ids.setdefault(actor_id, set()).add(edge.id)
            actor_evidence_totals[actor_id] = (
                actor_evidence_totals.get(actor_id, 0) + len(filtered_timestamps)
            )
            by_bucket = actor_buckets.setdefault(actor_id, {})
            for ts in filtered_timestamps:
                bucket_key = _bucket_start(ts, interval)
                if bucket_key not in by_bucket:
                    by_bucket[bucket_key] = {
                        "relation_ids": set(),
                        "incoming_ids": set(),
                        "outgoing_ids": set(),
                        "evidence_count": 0,
                        "predicates": Counter(),
                    }
                bucket = by_bucket[bucket_key]
                relation_ids = bucket["relation_ids"]
                incoming_ids = bucket["incoming_ids"]
                outgoing_ids = bucket["outgoing_ids"]
                predicates = bucket["predicates"]

                relation_ids.add(edge.id)
                if edge.subject_id == actor_id:
                    outgoing_ids.add(edge.id)
                if edge.object_id == actor_id:
                    incoming_ids.add(edge.id)
                bucket["evidence_count"] = int(bucket["evidence_count"]) + 1
                predicates[edge.predicate] += 1

    active_actor_ids = [actor_id for actor_id in threat_actor_ids if actor_id in actor_relation_ids]
    active_actor_ids.sort(
        key=lambda actor_id: (
            -len(actor_relation_ids.get(actor_id, set())),
            -actor_evidence_totals.get(actor_id, 0),
            node_by_id.get(actor_id).name.lower() if actor_id in node_by_id else actor_id,
        )
    )

    actors = []
    for actor_id in active_actor_ids[:top_n]:
        actor = node_by_id.get(actor_id)
        buckets = []
        for bucket_start in sorted(actor_buckets.get(actor_id, {})):
            data = actor_buckets[actor_id][bucket_start]
            predicate_counts = data["predicates"]
            top_predicates = [
                {"predicate": pred, "count": count}
                for pred, count in predicate_counts.most_common(10)
            ]
            buckets.append(
                {
                    "bucket_start": bucket_start.isoformat(),
                    "relation_count": len(data["relation_ids"]),
                    "incoming_relation_count": len(data["incoming_ids"]),
                    "outgoing_relation_count": len(data["outgoing_ids"]),
                    "evidence_count": int(data["evidence_count"]),
                    "top_predicates": top_predicates,
                }
            )

        actors.append(
            {
                "entity": {
                    "id": actor_id,
                    "name": actor.name if actor else actor_id,
                    "type": actor.type if actor else "threat_actor",
                },
                "relation_count": len(actor_relation_ids.get(actor_id, set())),
                "evidence_count": actor_evidence_totals.get(actor_id, 0),
                "bucket_count": len(buckets),
                "buckets": buckets,
            }
        )

    return {
        "seed": (
            {"id": seed_entity.id, "name": seed_entity.name, "type": seed_entity.type}
            if seed_entity
            else None
        ),
        "interval": interval,
        "depth": depth,
        "min_confidence": min_confidence,
        "source_uri": source_uri,
        "since": since_utc.isoformat() if since_utc else None,
        "until": until_utc.isoformat() if until_utc else None,
        "top_n": top_n,
        "actor_count_total": len(threat_actor_ids),
        "actor_count_active": len(active_actor_ids),
        "actor_count_returned": len(actors),
        "actors": actors,
    }


@router.post("/api/opencti/pull")
async def opencti_pull(
    entity_types: List[str] = Query(
        default=["Malware", "Threat-Actor", "Attack-Pattern", "Tool",
                 "Vulnerability", "Campaign", "Intrusion-Set",
                 "Indicator", "Infrastructure", "Course-Of-Action", "Report"]
    ),
    max_per_type: int = Query(default=0, ge=0, le=10000,
                              description="0 = fetch all (no limit)"),
):
    """Pull entities from OpenCTI as a background task."""
    client = _get_opencti_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OpenCTI not configured (set OPENCTI_URL and OPENCTI_TOKEN env vars)",
        )

    task = task_manager.create("opencti_pull", {
        "entity_types": entity_types,
        "max_per_type": max_per_type,
    })
    task_manager.update(task.id, status=TaskStatus.RUNNING, progress="Starting...")

    def _run_sync():
        try:
            result = pull_from_opencti(
                client, graph_store, entity_types, max_per_type,
                run_store=run_store, settings=settings,
                progress_cb=lambda msg: task_manager.update(task.id, progress=msg),
            )
            task_manager.update(
                task.id,
                status=TaskStatus.COMPLETED,
                progress=f"Done: {result.entities_pulled} entities, {result.relations_pulled} relations, {result.reports_queued} reports queued",
                detail={
                    "entities_pulled": result.entities_pulled,
                    "relations_pulled": result.relations_pulled,
                    "reports_queued": result.reports_queued,
                    "errors": result.errors[:20],
                },
                finished_at=datetime.utcnow().isoformat(),
            )
        except Exception as exc:
            task_manager.update(
                task.id,
                status=TaskStatus.FAILED,
                error=str(exc),
                finished_at=datetime.utcnow().isoformat(),
            )
        finally:
            client.close()

    async def _run():
        import asyncio
        await asyncio.to_thread(_run_sync)

    task_manager.start_async(task.id, _run())
    return {"task_id": task.id, "status": "running"}


@router.post("/api/scan")
async def scan_directory(
    extensions: str = Query(default=".txt,.md,.pdf,.json,.html,.csv,.xml,.yaml,.yml"),
):
    """Scan watched folders recursively and ingest all matching files as a background task."""
    import pathlib

    dirs = settings.watched_folders_list

    if not dirs:
        raise HTTPException(status_code=400, detail="No watched folders configured")

    # Verify at least one dir exists
    valid_dirs = [d for d in dirs if pathlib.Path(d).is_dir()]
    if not valid_dirs:
        raise HTTPException(status_code=400, detail=f"No watched folders found: {settings.watched_folders}")

    exts = set(e.strip().lower() for e in extensions.split(","))

    task = task_manager.create("filesystem_scan", {
        "watched_folders": valid_dirs,
    })
    task_manager.update(
        task.id,
        status=TaskStatus.RUNNING,
        progress="Discovering files...",
    )

    async def _run():
        import asyncio
        await asyncio.to_thread(_scan_files_sync, task.id, valid_dirs, exts)

    def _scan_files_sync(task_id: str, scan_dirs: List[str], extensions: set):
        import pathlib

        # Discover files in thread (can take a while for large dirs)
        task_manager.update(task_id, progress="Discovering files...")
        files: List[str] = []
        for d in scan_dirs:
            root = pathlib.Path(d)
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in extensions:
                    files.append(str(p))
        files.sort()

        if not files:
            task_manager.update(
                task_id,
                status=TaskStatus.COMPLETED,
                progress="No matching files found",
                finished_at=datetime.utcnow().isoformat(),
            )
            return

        task_manager.update(task_id, progress=f"Found {len(files)} files, starting ingestion...")

        from ..stix.importer import ingest_stix_bundle, parse_stix_file

        processed = 0
        stix_ok = 0
        queued = 0
        errors = []

        for filepath in files:
            processed += 1
            fname = os.path.basename(filepath)
            try:
                with open(filepath, "rb") as f:
                    raw = f.read()

                # Skip empty files
                if len(raw) < 10:
                    continue

                # STIX bundle?
                if filepath.lower().endswith(".json") and _is_stix_bundle(raw):
                    try:
                        bundle = parse_stix_file(raw, fname)
                        ingest_stix_bundle(
                            bundle, graph_store, source_uri=f"file://{filepath}"
                        )
                        stix_ok += 1
                    except Exception as exc:
                        errors.append(f"{fname}: STIX error: {exc}")
                    continue

                # Regular document -> LLM extraction queue
                text = _extract_text(raw, fname)
                if len(text.strip()) < 50:
                    continue

                source_uri = f"file://{filepath}"
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
                run_store.create_run(
                    run, source_uri, text,
                    {"filename": fname, "path": filepath, "size": len(raw)},
                )
                queued += 1

            except Exception as exc:
                errors.append(f"{fname}: {exc}")

            if processed % 25 == 0 or processed == len(files):
                task_manager.update(
                    task_id,
                    progress=f"Scanned {processed}/{len(files)}: {queued} queued, {stix_ok} STIX, {len(errors)} errors",
                )

        task_manager.update(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=f"Done: {queued} queued for LLM, {stix_ok} STIX imported, {len(errors)} errors",
            detail={
                "total_scanned": processed,
                "queued_for_llm": queued,
                "stix_imported": stix_ok,
                "errors": errors[:50],
            },
            finished_at=datetime.utcnow().isoformat(),
        )

    task_manager.start_async(task.id, _run())
    return {"task_id": task.id, "status": "running"}


@router.get("/api/tasks")
def list_tasks():
    """List all background tasks."""
    tasks = task_manager.list_all()
    return [{
        "id": t.id,
        "kind": t.kind,
        "status": t.status.value,
        "progress": t.progress,
        "started_at": t.started_at,
        "finished_at": t.finished_at,
        "error": t.error,
        "detail": t.detail,
    } for t in reversed(tasks)]


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    """Get status of a background task."""
    t = task_manager.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": t.id,
        "kind": t.kind,
        "status": t.status.value,
        "progress": t.progress,
        "started_at": t.started_at,
        "finished_at": t.finished_at,
        "error": t.error,
        "detail": t.detail,
    }


@router.post("/api/metrics/rollup")
async def trigger_metrics_rollup(
    lookback_days: int = Query(default=settings.metrics_rollup_lookback_days, ge=1, le=3650),
    min_confidence: float = Query(default=settings.metrics_rollup_min_confidence, ge=0.0, le=1.0),
):
    """Run daily threat-actor rollup in the background."""
    task = task_manager.create(
        "metrics_rollup",
        {
            "lookback_days": lookback_days,
            "min_confidence": min_confidence,
        },
    )
    task_manager.update(task.id, status=TaskStatus.RUNNING, progress="Starting metrics rollup...")

    def _run_sync():
        try:
            summary = metrics_store.rollup_daily_threat_actor_stats(
                lookback_days=lookback_days,
                min_confidence=min_confidence,
                source_uri=None,
            )
            task_manager.update(
                task.id,
                status=TaskStatus.COMPLETED,
                progress=(
                    "Done: "
                    f"{summary.get('docs_written', 0)} docs, "
                    f"{summary.get('buckets_written', 0)} buckets, "
                    f"{summary.get('actors_total', 0)} actors"
                ),
                detail=summary,
                finished_at=datetime.utcnow().isoformat(),
            )
        except Exception as exc:
            task_manager.update(
                task.id,
                status=TaskStatus.FAILED,
                error=str(exc),
                finished_at=datetime.utcnow().isoformat(),
            )

    async def _run():
        import asyncio
        await asyncio.to_thread(_run_sync)

    task_manager.start_async(task.id, _run())
    return {"task_id": task.id, "status": "running"}


@router.get("/api/stats")
def get_stats():
    """Quick graph stats with throughput."""
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    try:
        metrics = metrics_store.get_rollup_overview(days=30)
    except Exception:
        metrics = None
    return {
        "entities": graph_store.count_entities(),
        "relations": graph_store.count_relations(),
        "runs_total": run_store.count_runs(),
        "runs_pending": run_store.count_runs(status="pending"),
        "runs_running": run_store.count_runs(status="running"),
        "runs_completed": run_store.count_runs(status="completed"),
        "runs_failed": run_store.count_runs(status="failed"),
        "rate_per_hour": run_store.count_runs(status="completed", since=one_hour_ago),
        "metrics": metrics,
    }


@router.post("/api/recover-stale-runs")
def recover_stale_runs():
    """Reset runs stuck in 'running' back to 'pending'."""
    count = run_store.recover_stale_runs()
    return {"recovered": count}


@router.get("/api/watched-folders")
def get_watched_folders():
    """Return configured watched folders and their file counts."""
    import pathlib
    dirs = settings.watched_folders_list
    result = []
    for d in dirs:
        root = pathlib.Path(d)
        count = sum(1 for _ in root.rglob("*") if _.is_file()) if root.is_dir() else 0
        result.append({"path": d, "exists": root.is_dir(), "file_count": count})
    return result
