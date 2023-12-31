from asyncio import gather
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from server.lifespans.create_redis_index import create_redis_index
from server.lifespans.download_embeddings import download_embeddings
from server.lifespans.download_nltk import download_nltk
from server.lifespans.load_model import load_model


@asynccontextmanager
async def lifespans(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Summary
    -------
    the FastAPI lifespan function
    """
    print("Server has NOT started. Retrieving dependencies..")

    await gather(
        download_nltk(),
        download_embeddings(),
        create_redis_index(),
        load_model(),
    )

    print("Dependencies retrieved. Waiting for server initialisation..")

    yield
