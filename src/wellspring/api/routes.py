from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

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
from ..opencti.client import OpenCTIClient
from ..opencti.sync import pull_from_opencti
from ..storage.sqlite_store import SQLiteGraphStore, SQLiteRunStore
from .visualize import render_html
from .ui import render_root_ui
from .tasks import task_manager, TaskStatus

settings = get_settings()

graph_store = SQLiteGraphStore(settings.db_path)
run_store = SQLiteRunStore(settings.db_path)

router = APIRouter()


def _get_opencti_client() -> Optional[OpenCTIClient]:
    """Get an OpenCTI client if configured via env vars."""
    url = os.getenv("OPENCTI_URL")
    token = os.getenv("OPENCTI_TOKEN")
    if not url or not token:
        return None
    return OpenCTIClient(url, token)


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


@router.delete("/api/runs")
def delete_all_runs():
    """Delete all runs and associated documents/chunks."""
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    try:
        count = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM runs")
        conn.commit()
        return {"deleted": count}
    finally:
        conn.close()


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
    )
    title = f"Wellspring Graph: {seed_name or seed_id}"
    return render_html(subgraph, title=title)


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

    watched = os.environ.get("WATCHED_FOLDERS", "/data/documents")
    dirs = [d.strip() for d in watched.split(",") if d.strip()]

    if not dirs:
        raise HTTPException(status_code=400, detail="No watched folders configured")

    # Verify at least one dir exists
    valid_dirs = [d for d in dirs if pathlib.Path(d).is_dir()]
    if not valid_dirs:
        raise HTTPException(status_code=400, detail=f"No watched folders found: {watched}")

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


@router.get("/api/stats")
def get_stats():
    """Quick graph stats."""
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    try:
        entities = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
        relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
        runs = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
        pending = conn.execute("SELECT count(*) FROM runs WHERE status='pending'").fetchone()[0]
        running = conn.execute("SELECT count(*) FROM runs WHERE status='running'").fetchone()[0]
        completed = conn.execute("SELECT count(*) FROM runs WHERE status='completed'").fetchone()[0]
        return {
            "entities": entities,
            "relations": relations,
            "runs_total": runs,
            "runs_pending": pending,
            "runs_running": running,
            "runs_completed": completed,
        }
    finally:
        conn.close()


@router.get("/api/watched-folders")
def get_watched_folders():
    """Return configured watched folders and their file counts."""
    import pathlib
    watched = os.environ.get("WATCHED_FOLDERS", "/data/documents")
    dirs = [d.strip() for d in watched.split(",") if d.strip()]
    result = []
    for d in dirs:
        root = pathlib.Path(d)
        count = sum(1 for _ in root.rglob("*") if _.is_file()) if root.is_dir() else 0
        result.append({"path": d, "exists": root.is_dir(), "file_count": count})
    return result

