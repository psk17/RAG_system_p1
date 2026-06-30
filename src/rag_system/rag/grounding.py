def ensure_grounded(
    answer: str,
    contexts: list[str],
) -> str:
    if not contexts:
        return (
            "I cannot find the answer "
            "in the provided documents."
        )

    if not answer.strip():
        return (
            "I cannot find the answer "
            "in the provided documents."
        )

    return answer
