import re, unicodedata
import pandas as pd


def clean_text_for_semantics(text):
    """
    Prepare text for semantic similarity without destroying meaning.

    Steps:
    - Handle missing values
    - Normalize unicode (fix weird encodings)
    - Remove line breaks / tabs
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    - Lowercase (safe for most embedding models)

    IMPORTANT:
    - Does NOT remove punctuation
    - Does NOT remove stopwords
    - Does NOT stem/lemmatize
    """

    if pd.isna(text):
        return None

    text = str(text)

    # Normalize unicode (e.g., fancy quotes → standard)
    text = unicodedata.normalize("NFKC", text)

    # Replace line breaks and tabs with space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    text = text.strip()

    # Lowercase (safe for semantic models)
    text = text.lower()

    return text if text else None
