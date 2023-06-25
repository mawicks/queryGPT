import logging
import time

logger = logging.getLogger(__name__)


def backoff_and_retry(wrapped):
    for retry_count in range(5):
        try:
            result = wrapped()
            break
        except Exception as e:
            logger.warning(f"{wrapped.__name__} failed...")
            print(e)

            wait = 5 * 2**retry_count
            logger.warning(f"Retrying in {wait} seconds...")
            time.sleep(wait)
    else:
        raise RuntimeError("Too many retries.")

    return result
