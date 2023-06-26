import json
import logging

logger = logging.getLogger(__name__)

RELEVANT_DOCUMENT_COUNT = 100

from query_gpt.config import IRS990_SCHEMA
from query_gpt.completion import answer_question
from query_gpt.databases.qdrant import get_relevant_responses
from query_gpt.embeddings import embed_one


class QueryGPT:
    def __init__(self):
        """
        Construct answer bot from pre-computed vector search data.

        Arguments:
           data_dir: str - Path to directory containing vector_search_file
           vector_search_file: str - Filename of a pickle file
               containing a dictionary with two items:
                  'embeddings': list[list[float]] - embeddings
                  'docs': list[str] - List of documents whose embeddings are in the tree.
        """

    def get_answer(self, question):
        logger.info(f"Processing new question: {question}")
        logger.info("Getting embedding")
        embedding = embed_one(question)

        logger.info("Getting relevant responses")
        relevant_documents = get_relevant_responses(
            IRS990_SCHEMA, embedding, RELEVANT_DOCUMENT_COUNT
        )

        logger.info(f"Top 3 relevant documents:")
        for document in relevant_documents[:3]:
            logger.info(json.dumps(document))

        logger.debug("All relevant documents:")
        for doc_id, document in enumerate(relevant_documents):
            logger.debug(f"Record {doc_id}")
            logger.debug(json.dumps(document))

        logger.info("Calling OpenAI")

        def update_callback(partial_response):
            print(partial_response, end="")

        answer = answer_question(question, relevant_documents, update_callback)
        return answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    answer_bot = QueryGPT()

    while True:
        question = input("Question: ")
        if len(question) == 0:
            break
        answer = answer_bot.get_answer(question)
        print("\n")
