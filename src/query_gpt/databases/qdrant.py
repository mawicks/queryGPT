import logging
import os
from tqdm import tqdm
import json
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
CHUNK_SIZE = 100


def client_factory() -> QdrantClient:
    return QdrantClient(QDRANT_HOST, port=6333)


def remove_if_exists(schema: str):
    client = client_factory()

    if client.delete_collection(schema):
        logger.info(f"Removed existing schema: {schema}")

    logger.info(f"Creating new schema: {schema}")
    client.recreate_collection(
        schema,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        quantization_config=models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16,
                always_ram=True,
            ),
        ),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        on_disk_payload=True,
    )


def load_vectors(schema: str, docs: list[dict[str, str]], vectors: list[np.ndarray]):
    client = client_factory()

    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        points = []
        for doc, vector in zip(docs_chunk, vectors_chunk):
            text = json.dumps(doc)
            id = str(uuid.uuid3(uuid.NAMESPACE_OID, text))
            points.append(
                models.PointStruct(id=id, vector=vector.tolist(), payload=doc)
            )

        client.upsert(schema, points=points)


def get_relevant_responses(schema, embedding, count):
    client = client_factory()
    logger.info(f"Querying relevant verbatim responses...")

    query_result = client.search(schema, embedding, limit=count)
    relevant_documents = [point.payload for point in query_result]
    return relevant_documents
