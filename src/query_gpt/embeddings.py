import gc
import logging
import os
import time
from typing import Callable

import openai
import pandas as pd
from tqdm import tqdm

from query_gpt.retry import backoff_and_retry

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_SEARCH_FILE_TEMPLATE = (
    "irs_form_990_embeddings_{year}_{segment}_{chunk_id}.parquet"
)

# How many embeddings to store in a single file?
# We limit it because the chunks in these files get generated in memory and
# read back into memory.

CHUNK_SIZE = 5_000

# How many documents to pass to OpenAPI at one time.
OPENAI_BATCH_SIZE = 100


def embed_chunk(chunk_of_docs, doc_to_string) -> dict[str, object]:
    """
    Use OpenAI to Compute the embeddings for the documents in `docs`
    and return a dictionary containing the documents that were embedded
    along with the associated embeddings.

    Arguments:
        chunk_of_docs: list[dict] - Chunk of docs to embed
        doc_to_string: Callable[[dict], str] - Function that converts a document
            dict to an embeddable string
    Returns:
        A dictionary with two keys: "docs" and "embeddings".
        The value of "doc" is the original `chunk_of_docs`
        The value of "embedding" is a list of embeddigns matching the list of `chunk_of_docs`
    """

    embeddings = []
    usage = 0
    for batch_index in tqdm(range(0, len(chunk_of_docs), OPENAI_BATCH_SIZE)):
        # Convert list of documents to embeddable text
        batch = list(
            map(
                doc_to_string,
                chunk_of_docs[batch_index : batch_index + OPENAI_BATCH_SIZE],
            )
        )

        # OpenAI calls can fail.  Wrap in a retry loop  Try the batch up to
        # RETRY_LIMIT times

        def try_once():
            result = openai.Embedding.create(
                input=batch,
                model=EMBEDDING_MODEL,
            )
            return result

        result = backoff_and_retry(try_once)

        embeddings.extend([e["embedding"] for e in result["data"]])  # type:ignore
        usage += result["usage"]["total_tokens"]  # type:ignore

    logger.info(f"Total tokens used: {usage}")

    assert len(embeddings) == len(chunk_of_docs)

    search_data = {
        "doc": chunk_of_docs,
        "embedding": embeddings,
    }

    return search_data


def embed_one(text: str):
    api_result = openai.Embedding.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return api_result["data"][0]["embedding"]  # type:ignore


def compute_search_embeddings(
    docs: list[dict],
    doc_to_string: Callable[[dict], str],
    data_dir: str,
    year: int,
    segment: str,
    limit: int | None = None,
):
    """
    Compute the embeddings for a list of documents in `docs` and store
    them as files in `data_dir` use the filename template
    VECTOR_SEARCH_FILE_TEMPLATE which is parametered by the passed `segment`
    and an internally generated `chunk_id` if the document list needs
    to be broken up to process it.

    Arguments:
        docs: list[dict] - List of document dictionaries
        doc_to_string: Callable[[dict], str] - Function that converts a document
            dict to an embeddable string
        data_dir: str - Path in which to write output files
        segment: str - Segment ID to use in the file name
        year: int - Year filed?

    Returns:
        None
    """

    # Don't process all the documents if a limit was specified (useful for
    # computing a partial set of embeddings for testing).
    docs_to_embed = docs[:limit]

    # Break the documents into chunks because holding a full set of embeddings
    # in memory can exhaust resources and cause the process to be killed in
    # contrained environments.

    logger.info(f"Embedding {len(docs_to_embed):,d} documents")

    for chunk_id, chunk_index in enumerate(
        tqdm(range(0, len(docs_to_embed), CHUNK_SIZE))
    ):
        search_data = embed_chunk(
            docs_to_embed[chunk_index : chunk_index + CHUNK_SIZE], doc_to_string
        )
        df = pd.DataFrame(data=search_data)

        vector_search_filename = VECTOR_SEARCH_FILE_TEMPLATE.format(
            year=year, segment=segment, chunk_id=chunk_id
        )
        df.to_parquet(os.path.join(data_dir, vector_search_filename))

        # As mentioned above, instantiating embeddings in memory consumes it.
        # Help out the garbage collector by telling it when we're completely
        # finished with a batch
        gc.collect()


if __name__ == "__main__":
    x = embed_one("this is a test")
    print(x)
    print("done")
