import textstat


def score_readability(text):
    """
    Compute readability metrics for a personal statement.

    Parameters:
        - text (str): The full personal statement text.

    Returns:
        - dict: A dictionary containing readability scores.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "smog_index": None,
            "automated_readability_index": None,
            "dale_chall_score": None
        }

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text)
    }
