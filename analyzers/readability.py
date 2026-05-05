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
            "gunning_fog_index": None,
            "linsear_write_formula": None
        }

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "gunning_fog_index": textstat.gunning_fog(text),
        "linsear_write_formula": textstat.linsear_write_formula(text)
    }
