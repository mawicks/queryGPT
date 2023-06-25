import os
from tqdm import tqdm
import json
from query_gpt.load_vector_db import CHUNK_SIZE, logger

import weaviate

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

def client_factory():
    return weaviate.Client(url=WEAVIATE_URL)


def remove_if_exists_weaviate(schema: str):
    client = client_factory()

    if client.schema.exists(schema):
        logger.info(f"Removing existing {schema} Weaviate class")
        client.schema.delete_class(schema)

    logger.info(f"Creating Weaviate class for {schema}")
    class_obj = {
        "class": schema,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "vectorCacheMaxObjects": 25_000,
        },
    }
    client.schema.create_class(class_obj)


def load_weaviate(
    schema: str, docs: list[str], vectors: list[list[float]]
):
    client = client_factory()
    
    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        with client.batch() as batch:
            for doc, vector in zip(docs_chunk, vectors_chunk):
                x = json.dumps(doc)
                object = {"doc": json.dumps(doc)}
                batch.add_data_object(
                    data_object=object, class_name=schema, vector=vector
                )