import gc
from glob import glob
import logging
import os
import random

import click
import pandas as pd
from tqdm import tqdm

from query_gpt.config import DATA_DIR, IRS990_SCHEMA

# from query_gpt.databases.weaviate import load_weaviate, remove_if_exists_weaviate
from query_gpt.databases.qdrant import load_vectors, remove_if_exists

CHUNK_SIZE = 50
FILENAME_TEMPLATE = "irs_form_990_embeddings*.parquet"

FILE_LIMIT_QUICK = 10
RECORD_LIMIT_QUICK = 500

logger = logging.getLogger("query_gpt")


@click.command
@click.option(
    "--full", is_flag=True, help="Load the full dataset (default is partial dataset"
)
def load_vector_db_command(full):
    random_state = random.Random(42)

    remove_if_exists(IRS990_SCHEMA)

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

        docs = list(search_data["doc"])
        data = list(search_data["embedding"])

        load_vectors(IRS990_SCHEMA, docs, data)

        # Try to free up some memory
        del docs, data, search_data
        gc.collect()

    logger.info("done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Raise the log level for the 'httpx' logger.
    # We don't need to see HTTP return codes.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_vector_db_command()
