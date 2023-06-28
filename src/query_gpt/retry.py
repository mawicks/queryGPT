import logging
import time

logger = logging.getLogger(__name__)


def backoff_and_retry(wrapped, max_attempts=5, initial_wait_seconds=5):
    for retry_count in range(max_attempts):
        try:
            result = wrapped()
            break
        except Exception as e:
            logger.warning(f"{wrapped.__name__} failed...")
            logger.warning(e)

            if retry_count + 1 < max_attempts:
                wait = initial_wait_seconds * 2**retry_count
                logger.warning(f"Retrying in {wait} seconds...")
                time.sleep(wait)
    else:
        raise RuntimeError("Too many retries.")

    return result
