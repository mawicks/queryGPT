import logging
import os
from tqdm import tqdm
import json

import weaviate

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
CHUNK_SIZE = 50

logger = logging.getLogger(__name__)


def client_factory():
    return weaviate.Client(url=WEAVIATE_URL)


def remove_if_exists(schema: str):
    client = client_factory()

    if client.schema.exists(schema):
        logger.info(f"Removing existing Weaviate class: {schema}")
        client.schema.delete_class(schema)

    logger.info(f"Creating Weaviate class: {schema}")
    class_obj = {
        "class": schema,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "vectorCacheMaxObjects": 25_000,
        },
    }
    client.schema.create_class(class_obj)


def load_vectors(schema: str, docs: list[str], vectors: list[list[float]]):
    client = client_factory()

    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        with client.batch() as batch:
            for doc, vector in zip(docs_chunk, vectors_chunk):
                object = {"doc": json.dumps(doc)}
                batch.add_data_object(
                    data_object=object, class_name=schema, vector=vector
                )
