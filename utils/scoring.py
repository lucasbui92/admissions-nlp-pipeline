import nltk, textstat, language_tool_python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def score_tfidf_embedding_cosine(statement, description, model=get_embedding_model(), alpha=0.7):
    """
    Compute a weighted cosine similarity score combining TF-IDF (keyword overlap)
    and sentence embedding (semantic meaning).

    Parameters:
        statement (str): Cleaned personal statement text.
        description (str): Cleaned course description text.
        model: SentenceTransformer model instance (loaded once via default argument).
        alpha (float): Weight given to the embedding score (default 0.7).
                       TF-IDF weight = 1 - alpha.

    Returns:
        float: Weighted combined score in [0, 1], or None if inputs are invalid.
    """
    if not statement or not description:
        return None

    corpus = [statement, description]
    tfidf_matrix = TfidfVectorizer().fit_transform(corpus)
    tfidf_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0])

    embeddings = model.encode(corpus, convert_to_tensor=True)
    embedding_score = float(cos_sim(embeddings[0], embeddings[1]))

    return round(alpha * embedding_score + (1 - alpha) * tfidf_score, 4)

def get_language_tool():
    return language_tool_python.LanguageTool("en-UK")

def trim_text_by_words(text, max_words=30):
    """
    Generate a shortened preview of a long personal statement for output storage
    and visual inspection. The full text should still be used for grammar analysis.

    Parameters
        text (str): Full personal statement text.
        max_words (int, optional): Maximum number of words to keep in the preview (default = 30).

    Returns
        str: Trimmed text preview. If the original text exceeds the word limit, an ellipsis ("...") is appended.
    """
    if not isinstance(text, str):
        return ""

    words = text.strip().split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def score_grammar_quality(text, tool):
    """
    Evaluate grammatical quality using LanguageTool and compute
    a document-level grammar score.

    Returns
        dict with:
            final_score
            error_count
            word_count
            char_count
            matches
    """
    # Handle empty text
    if not text or not isinstance(text, str):
        return {
            "final_score": None,
            "error_count": 0,
            "word_count": 0,
            "char_count": 0,
            "matches": []
        }

    matches = tool.check(text)

    word_count = len(text.split())
    char_count = len(text)   # character count including spaces
    error_count = len(matches)

    # Grammar score based on word-normalised error proportion
    error_ratio = 0 if word_count == 0 else error_count / word_count
    final_score = 100 * (1 - error_ratio)

    match_list = []
    for m in matches:
        item = {
            "message": m.message,
            "context": m.context,
            "category": getattr(m.category, "id", str(m.category))
        }
        match_list.append(item)

    return {
        "final_score": round(final_score, 2),
        "error_count": error_count,
        "word_count": word_count,
        "char_count": char_count,
        "matches": match_list
    }

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
