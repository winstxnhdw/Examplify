FROM python:slim AS python-builder

WORKDIR /home/user

ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python
ENV UV_COMPILE_BYTECODE=1
ENV UV_LOCKED=1
ENV UV_NO_DEV=1
ENV UV_NO_EDITABLE=1
ENV PYTHONOPTIMIZE=2

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=server/pyproject.toml,target=server/pyproject.toml \
    uv sync --no-install-project --package server

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv uv sync --package server


FROM curlimages/curl:latest AS curl-builder

RUN curl -O https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/eng.traineddata


FROM python:slim

ENV HOME=/home/user
ENV PATH=$HOME/.venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TESSDATA_PREFIX=/usr/share/tessdata

WORKDIR $HOME

COPY --from=curl-builder   /home/curl_user/eng.traineddata $TESSDATA_PREFIX/eng.traineddata
COPY --chown=user --from=python-builder $HOME/.venv .venv

CMD ["examplify"]
