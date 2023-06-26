import logging
import time

import openai
import tiktoken
from query_gpt.config import MODEL

from query_gpt.irs_data import make_prompt

MAX_COMPLETION_TOKENS = 1000
CHUNK_INTERVAL = 16  # Update interval while streaming.
MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)

def openai_completion(messages, update_callback=None):
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        presence_penalty=1.0,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=0.01,  # OpenAI says that one of temperature or top_p should be 1.0
        top_p=0.8,
        stream=True,
    )

    completion = ""
    chunk_time = 0.0
    chunk = None

    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk

        chunk_message = chunk["choices"][0]["delta"].get(
            "content", ""
        )  # extract the message

        completion += chunk_message
        if update_callback:
            update_callback(chunk_message)

    # print the time delay and text received
    logger.info(f"Full response received {chunk_time:.2f} seconds after request")
    return chunk, completion


def answer_question(question, context, update_callback=None, max_attempts=MAX_ATTEMPTS):
    """
    Given a question and context (i.e., list of survey responses),
    query openai for an answer to the question.
    Make up to `max_attempts` to get a satisfactory answer.  By "satisfactory",
    we just mean that the model didn't get cut off in mid sentence.

    Arguments:
        question: str - Question about the survey data
        context: list[str] - Survey responses to provide as context
        max_attempts: int - Maximum number of API calls to attempt.

    """
    result = None
    failures = 0
    for _ in range(max_attempts):
        prompt = make_prompt(question, context, failures)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        completion, message = openai_completion(messages, update_callback)

        response = completion.choices[0]
        if response.finish_reason == "stop":
            result = message.strip()
            # usage = completion.usage
            break
        else:
            logger.warning(f"Finish reason: {response.finish_reason}")
            logger.warning(f"Truncated answer: {message.strip()}")
            logger.warning("Trying again with a different prompt...")
            failures += 1

    return result
