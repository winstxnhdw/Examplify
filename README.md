# Examplify

> STILL WIP

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![main.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/main.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/main.yml)
[![Dockerise](https://github.com/winstxnhdw/Examplify/actions/workflows/docker.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/docker.yml)
[![formatter.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/formatter.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/formatter.yml)
[![dependabot.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/dependabot.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/dependabot.yml)

<div align="center">
    <img src="resources/logo.png" width="70%" />
</div>

`Examplify` is an offline CPU-first low-resource chat application to perform Retrieval-Augmented Generation (RAG) on your corpus of data. It utilises an 8-bit quantised openchat-3.6 model, running on CTranslate2's inference engine for maximum CPU performance.

## Requirements

- [Docker Compose](https://docs.docker.com/compose/install/)
- 10 GB RAM

## Benchmarks

| Model                                                                                | Tokens | Time (s)  | Throughput (t/s) | Device             |
| ------------------------------------------------------------------------------------ | ------ | --------- |----------------- | ------------------ |
| [zephyr-7b-beta-ct2-int8](https://huggingface.co/winstxnhdw/zephyr-7b-beta-ct2-int8) | 219    | 2.272     | 96.396           | NVIDIA RTX 3090    |
| [zephyr-7b-beta-ct2-int8](https://huggingface.co/winstxnhdw/zephyr-7b-beta-ct2-int8) | 211    | 24.482    | 8.619            | Intel i7-8700      |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 151    | 0.832     | 181.469          | NVIDIA RTX 3090    |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 156    | 1.573     | 99.160           | NVIDIA RTX 3080 Ti |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 152    | 10.611    | 14.325           | Intel i7-12800H    |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 151    | 9.696     | 15.574           | Intel i7-8700      |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 151    | 9.667     | 15.620           | Intel i7-1260P     |
| [openchat-3.5-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.5-ct2-int8)     | 151    | 20.794    | 7.262            | Intel i9-11900H    |
| [openchat-3.6-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.6-ct2-int8)     | 174    | 1.340     | 129.828          | NVIDIA RTX 3090    |
| [openchat-3.6-ct2-int8](https://huggingface.co/winstxnhdw/openchat-3.6-ct2-int8)     | 189    | 22.500    | 8.400            | Intel i7-8700      |

## Setup

To setup the application, we must populate your `.env` file. You can do this with the following.

> [!IMPORTANT]\
> `OMP_NUM_THREADS` should correspond to the number of physical cores available.

```bash
{
  echo BACKEND_URL=localhost
  echo BACKEND_PORT=443
  echo CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
  echo OMP_NUM_THREADS=8
} > .env
```

## Usage

You can start the application and access the Swagger UI at [https://localhost/api/schema/swagger](https://localhost/api/schema/swagger).

> [!WARNING]\
> Before offline usage, you must run the application at least once with internet access to install any necessary dependencies.

```bash
make u
```

## Development

Install all dependencies with the following.

```bash
poetry install
```

Delete cached models.

```bash
sudo make clean
```
