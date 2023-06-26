import logging
import time

import numpy as np
import openai
import tiktoken

MODEL = "gpt-3.5-turbo-16k"  # Or use "text-davinci-003" for GPT-3

# The token counts returned by tiktoken don't exctly match the actual token
# counts in the API Allow a little margin of error to stay below the 16k limit.
# (Tiktoken typically underestimates the token count by eight.)
INPUT_TOKEN_GOAL = 14_900
MAX_COMPLETION_TOKENS = 1000
CHUNK_INTERVAL = 16  # Update interval while streaming.
MAX_ATTEMPTS = 3

# Encoder to use to pre-check the token count
ENCODER = tiktoken.encoding_for_model(MODEL)

logger = logging.getLogger(__name__)

from query_gpt.data import doc_to_string


def make_prompt(question: str, items: list[dict[str, str]], failures: int) -> str:
    """
    Given a question about the survey data, design a prompt for
    openai that should produce an answer to the question.
    The context is the additional information to provide.

    Arguments:
        question: str - Question about the survey data
        items: list[dict[str,str]] - Documents to be queried
        failures: int - Number of API calls that have failed because of reponse length.
    """

    prefix = (
        "The following records contain information taken from tax records "
        "for non-profit organizations operating in the US. "
        "The data for a single organization is delimited by <record> and </record>. "
        "At the end of these records, there is a question for you to answer about "
        "these non-profit organizations."
        "Try to keep the total response below 500 words.\n"
    )

    instruction = (
        "Please answer the question below about non-profit organizations.  "
        "Some of the records above may not be relevant to the question.  Pleas ignore "
        "any irrelevant records.  The most relevant ones may be near the top of the list. "
        "Remember to keep "
        "the response below 500 words. If you are asked to provide a list, you may "
        "need to omit some items from the list.  "
        "If so, state the the list is represntative and not complete. "
        "Capitalize any responses appropriately, even if the source data was presented in ALL CAPS. "
        f"{'Be *EXTREMELY* BRIEF in your answer. ' if failures > 0 else ''}"
        "Answer the question precisely and exclude any records that are not relevant to the question. "
        "The answer should be responsive. "
        " It's better to provide no response than to provide a response with irrelevant information. "
        "Base your answer primarily on the records above, but you may fill in "
        "holes based on any prior knowledge you have of these organizations.\n"
        f"Question: {question}\n"
        "Answer: "
    )
    formatted_items = [f"<record>\n{doc_to_string(item)}</record>\n" for item in items]

    # When counting fixed strings, two for the separators we'll add later.
    fixed_count = len(ENCODER.encode(prefix)) + len(ENCODER.encode(instruction)) + 2

    # When counting item tokens, add one for the separator we'll add later.
    variable_counts = [len(ENCODER.encode(item)) + 1 for item in formatted_items]
    allowed_item_count = sum(
        np.cumsum(variable_counts) < (INPUT_TOKEN_GOAL / (2**failures) - fixed_count)
    )

    context = "\n".join(formatted_items[:allowed_item_count])
    prompt = f"{prefix}\n{context}\n{instruction}"
    token_count = len(ENCODER.encode(prompt))
    logger.info(f"tiktoken token estimate: {token_count}")
    return prompt


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
