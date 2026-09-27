# Examplify

## Requirements

- Linux with an NVIDIA GPU and driver compatible with CUDA 12.8
- Docker Engine with NVIDIA Container Toolkit
- OpenCode
- At least 40 GB of free Docker storage

## Start

LightRAG opens at <http://127.0.0.1:9621> and TabbyAPI at <http://127.0.0.1:5000>.

```bash
docker compose up
```

The OpenAI-compatible LightRAG endpoint is `http://127.0.0.1:4000/v1`:

- Model: `lightrag`
- API key: `sk-local-lightrag`

Then, from another terminal:

```bash
opencode
```
