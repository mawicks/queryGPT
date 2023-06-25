import gc
from glob import glob
import logging
import os
import random

import click
import pandas as pd
from tqdm import tqdm

from query_gpt.config import DATA_DIR
from query_gpt.weaviate import load_weaviate, remove_if_exists_weaviate


CHUNK_SIZE = 50

FILENAME_TEMPLATE = "irs_form_990_embeddings*.parquet"

FILE_LIMIT_QUICK = 10
RECORD_LIMIT_QUICK = 500

logger = logging.getLogger(__name__)


@click.command
@click.option(
    "--full", is_flag=True, help="Load the full dataset (default is partial dataset"
)
def load_weaviate_command(full):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random_state = random.Random(42)

    remove_if_exists_weaviate("irs990")

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

        load_weaviate("irs990", docs, data)

        # Try to free up some memory
        del docs, data, search_data
        gc.collect()

    logger.info("done")

if __name__ == "__main__":
    load_weaviate_command()
