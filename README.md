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

## Use OpenAI for query answers

`Examplify` is meant to work offline, but you can route query answers to OpenAI instead of TabbyAPI by running the following. The online query uses `gpt-6-astra` with OpenAI Fast mode, which has higher per-token pricing than Standard processing.

```bash
docker compose -f compose.yaml -f compose.online.yaml up
```
