# Examplify

> STILL WIP

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![main.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/main.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/main.yml)
[![formatter.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/formatter.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/formatter.yml)
[![dependabot.yml](https://github.com/winstxnhdw/Examplify/actions/workflows/dependabot.yml/badge.svg)](https://github.com/winstxnhdw/Examplify/actions/workflows/dependabot.yml)

`Examplify` is an offline CPU-first memory-scarce chat application to perform Retrieval-Augmented Generation (RAG) on your corpus of data. It utilises an 8-bit quantised zephyr-7b-beta model, running on CTranslate2's inference engine for maximum CPU performance.

## Requirements

- [Docker Compose](https://docs.docker.com/compose/install/)
- 10 GB RAM

## Setup

To setup the application, we must populate your `.env` file. You can do this with the following.

> [!IMPORTANT]\
> `OMP_NUM_THREADS` should correspond to the number of physical cores available.

```bash
{
  echo BACKEND_URL=http://localhost
  echo BACKEND_PORT=5000
  echo CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
  echo OMP_NUM_THREADS=8
} > .env
```

## Usage

You can start the application and access the Swagger UI at [http://localhost:5000/api/docs](http://localhost:5000/api/docs).

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
