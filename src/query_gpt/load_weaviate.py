import gc
from glob import glob
import json
import logging
import os
import random
import time

import click
import pandas as pd
from tqdm import tqdm
import weaviate

from query_gpt.config import DATA_DIR
from query_gpt.retry import backoff_and_retry

CHUNK_SIZE = 50

FILENAME_TEMPLATE = "irs_form_990_embeddings*.parquet"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

FILE_LIMIT_QUICK = 10
RECORD_LIMIT_QUICK = 500

logger = logging.getLogger(__name__)


def remove_if_exists(schema: str):
    client = weaviate.Client(url=WEAVIATE_URL)

    if client.schema.exists(schema):
        logger.info(f"Removing existing {schema} Weaviate class")
        client.schema.delete_class(schema)

    logger.info(f"Creating Weaviate class for {schema}")
    class_obj = {
        "class": schema,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "vectorCacheMaxObjects": 1_000,
        },
    }
    client.schema.create_class(class_obj)


def load_weaviate(schema: str, docs: list[str], vectors: list[list[float]]):
    for chunk in tqdm(range(0, len(docs), CHUNK_SIZE)):
        docs_chunk = docs[chunk : chunk + CHUNK_SIZE]
        vectors_chunk = vectors[chunk : chunk + CHUNK_SIZE]

        def try_once():
            client = weaviate.Client(url=WEAVIATE_URL)
            with client.batch() as batch:
                for doc, vector in zip(docs_chunk, vectors_chunk):
                    object = {"doc": json.dumps(doc)}
                    batch.add_data_object(
                        data_object=object, class_name=schema, vector=vector
                    )

        backoff_and_retry(try_once)

        gc.collect()


@click.command
@click.option(
    "--full",
    is_flag=True,
    help="Load the full dataset (default loads only a partial dataset)",
)
def load_weaviate_command(full):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random_state = random.Random(42)

    remove_if_exists("irs990")

    filenames = glob(os.path.join(DATA_DIR, "embeddings", FILENAME_TEMPLATE))

    if not full:
        file_limit = min(FILE_LIMIT_QUICK, len(filenames))
        filenames = random_state.sample(filenames, file_limit)

    logger.info(f"Loading embedding data ({len(filenames):,d} files) to Weaviate")

    for filename in tqdm(filenames):
        search_data = pd.read_parquet(filename)
        if not full:
            record_limit = min(RECORD_LIMIT_QUICK, len(search_data))
            search_data = search_data.sample(n=record_limit, random_state=42)

        docs = search_data["doc"]
        data = search_data["embedding"]

        load_weaviate("irs990", docs, data)

        # Try to free up some memory
        del docs, data, search_data
        gc.collect()

    logger.info("done")


if __name__ == "__main__":
    load_weaviate_command()
