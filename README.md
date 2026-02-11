# Wellspring

LLM-Native Knowledge Graph (KG) Platform. This repo contains a Python 3.11+ implementation that ingests unstructured text, extracts subject-predicate-object triples with Ollama, normalizes and deduplicates entities, stores a persistent knowledge graph with provenance, and exposes APIs for query, explanation, and visualization.

## Quickstart

1) Start services:

```bash
docker compose up --build
```

2) Ingest text:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_uri": "local://demo", "text": "Ada Lovelace wrote notes on the Analytical Engine."}'
```

3) Query:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"seed_name": "Ada Lovelace", "depth": 1}'
```

4) Visualize:

Open `http://localhost:8000/visualize?seed_name=Ada%20Lovelace&depth=1`

## Configuration

Environment variables:
- `OLLAMA_BASE_URL` (default: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL` (default: `llama3.1`)
- `DB_PATH` (default: `/data/wellspring.db`)
- `CHUNK_SIZE` (default: `1200`)
- `CHUNK_OVERLAP` (default: `200`)
- `PROMPT_VERSION` (default: `v1`)
- `LOG_LEVEL` (default: `INFO`)
- `ENABLE_COOCCURRENCE` (default: `0`)
- `CO_OCCURRENCE_MAX_ENTITIES` (default: `25`)
- `ENABLE_INFERENCE` (default: `0`)

Inference (when enabled) currently applies a simple transitive rule for `is_a` relations within a chunk.
