# Examplify

## Requirements

- Linux with an NVIDIA GPU and driver compatible with CUDA 12.8
- Docker Engine with NVIDIA Container Toolkit
- OpenCode
- At least 40 GB of free Docker storage

## Start

LightRAG opens at <http://127.0.0.1:9621> and TabbyAPI at <http://127.0.0.1:5000>.
Set `OPENAI_API_KEY` in `.env` before starting; LightRAG uses it for document extraction.

```bash
docker compose up
```

The OpenAI-compatible LightRAG endpoint is `http://127.0.0.1:4000/v1` (model `lightrag`).

Then, from another terminal:

```bash
opencode
```

The CPU embedding model can take several minutes to warm up. Compose waits for
its health check before starting LightRAG. If an earlier indexing run left
documents in `failed` status, retry them after the services are healthy:

```bash
curl -X POST http://127.0.0.1:9621/documents/reprocess_failed
curl http://127.0.0.1:9621/documents/status_counts
```

## Indexing throughput

LightRAG processes up to three documents at once. Extraction requests to the
OpenAI model may run eight at once; query and keyword requests remain limited
to one each. The local CPU embedding model is limited to one request per
embedding worker and four texts per request. Apply changes to these settings
only after an indexing batch finishes:

```bash
docker compose up -d --no-build lightrag
```

The embedding model can still dominate indexing time on a CPU. Changing the
embedding model requires rebuilding the existing vector index.
