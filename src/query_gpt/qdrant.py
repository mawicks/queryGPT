import os
from tqdm import tqdm
import json
from query_gpt.load_vector_db import CHUNK_SIZE, logger

from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")


def client_factory() -> QdrantClient:
    return QdrantClient(QDRANT_HOST, port=6333)


def remove_if_exists_weaviate(schema: str):
    client = client_factory()

    if schema in list(client.get_collections()):
        logger.info(f"Removing existing {schema} Weaviate class")
        client.delete_collection(schema)

    logger.info(f"Creating Weaviate class for {schema}")
    client.create_collection(
        schema,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        quantization_config=models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16,
                always_ram=True,
            ),
        ),
    )

def load_weaviate(schema: str, docs: list[str], vectors: list[list[float]]):
    client = client_factory()

    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        for doc, vector in zip(docs_chunk, vectors_chunk):
            x = json.dumps(doc)
            object = {"doc": json.dumps(doc)}
            # Do something
