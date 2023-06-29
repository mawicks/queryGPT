import logging
import os
from tqdm import tqdm
import json
import uuid
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from query_gpt.retry import backoff_and_retry

logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
CHUNK_SIZE = 100
DEFAULT_READY_POLL_TIME_SECONDS = 60


def client_factory() -> QdrantClient:
    return QdrantClient(QDRANT_HOST, port=6333)


def remove_and_recreate_schema(schema: str):
    client = client_factory()

    if client.delete_collection(schema):
        logger.info(f"Removed existing schema: {schema}")

    logger.info(f"Creating new schema: {schema}")

    __quantization_config__ = models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X16,
            always_ram=True,
        )
    )
    # Setting indexing_threshold to 0 during large uploads is
    # recommended here: https://qdrant.tech/documentation/tutorials/bulk-upload/
    client.recreate_collection(
        schema,
        vectors_config=models.VectorParams(
            size=1536, distance=models.Distance.COSINE, on_disk=True
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        hnsw_config=models.HnswConfigDiff(
            on_disk=True,
        ),
        on_disk_payload=True,
    )


def restore_indexing(schema: str):
    client = client_factory()
    client.update_collection(
        schema,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20_000,
            memmap_threshold=20_000,
        ),
    )


def wait_until_ready(schema: str):
    wait = DEFAULT_READY_POLL_TIME_SECONDS
    logger.info(f"Waiting {wait} seconds before polling {schema} for status")

    client = client_factory()
    while True:
        time.sleep(wait)
        collection = client.get_collection(schema)
        if collection.status == "green":
            logger.info(f"Collection {schema} is ready.")
            break
        else:
            logger.info(f"Collection {schema} is not ready.  Waiting {wait} seconds...")
    return


def rename(loading_collection: str, collection: str):
    client = client_factory()

    # Remove the alias if it already exists and add the alias to
    # the loading collection.

    result = client.update_collection_aliases(
        [
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=collection)
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=loading_collection, alias_name=collection
                )
            ),
        ]
    )

    if not result:
        raise RuntimeError("update_collection_aliases failed")

    # Remove any collections left over from the past that resemble this
    # collectio name but are not the loading collection.
    for collection_description in client.get_collections().collections:
        existing_collection = collection_description.name
        if (
            existing_collection.startswith(collection)
            and existing_collection != loading_collection
        ):
            logger.info(f"Removing old collection: {existing_collection}")
            client.delete_collection(existing_collection)


def load_vectors(schema: str, docs: list[dict[str, str]], vectors: list[np.ndarray]):
    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        points = []
        for doc, vector in zip(docs_chunk, vectors_chunk):
            # Sort the keys to guarantee uniqueness of the `id`.
            text = json.dumps(doc, sort_keys=True)
            id = str(uuid.uuid3(uuid.NAMESPACE_OID, text))
            points.append(
                models.PointStruct(id=id, vector=vector.tolist(), payload=doc)
            )

        def try_once():
            client = client_factory()
            client.upsert(schema, points=points)

        backoff_and_retry(try_once)


def get_relevant_responses(schema, embedding, count):
    client = client_factory()
    logger.info(f"Querying relevant verbatim responses...")

    query_result = client.search(schema, embedding, limit=count)
    relevant_documents = [point.payload for point in query_result]
    return relevant_documents
