import time


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3, sleep_seconds: int = 10):
    """
    Invokes a LangChain chain with automatic retry on 429 (rate limit) errors.
    On any other exception the error is re-raised immediately.

    Args:
        chain:         A compiled LangChain runnable (prompt | llm | parser, etc.)
        inputs:        dict of template variables to pass to chain.invoke()
        max_retries:   maximum number of total attempts (default 3)
        sleep_seconds: seconds to sleep between retries when rate-limited (default 10)

    Returns:
        The result of chain.invoke(inputs) on the first successful attempt.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            err = str(exc).lower()
            is_rate_limit = (
                "429" in str(exc)
                or "resource exhausted" in err
                or "quota" in err
                or "rate limit" in err
            )
            if is_rate_limit and attempt < max_retries:
                print(
                    f"[Rate limit 429] Attempt {attempt}/{max_retries} hit quota. "
                    f"Sleeping {sleep_seconds}s before retry…"
                )
                time.sleep(sleep_seconds)
            else:
                raise
