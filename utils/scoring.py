import nltk, textstat, language_tool_python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def score_document_level_similarity(statement, description, alpha=0.7, desc_embedding=None, stmt_embedding=None):
    """
    Compute a weighted cosine similarity score combining TF-IDF (keyword overlap)
    and sentence embedding (semantic meaning).

    Parameters:
        statement (str): Cleaned personal statement text.
        description (str): Cleaned course description text.
        alpha (float): Weight given to the embedding score (default 0.7).
                       TF-IDF weight = 1 - alpha.
        desc_embedding: Pre-computed embedding for description. Encoded on-the-fly if None.
        stmt_embedding: Pre-computed embedding for statement. Encoded on-the-fly if None.

    Returns:
        float: Weighted combined score in [0, 1], or None if inputs are invalid.
    """
    if not statement or not description:
        return None

    tfidf_matrix = TfidfVectorizer().fit_transform([statement, description])
    tfidf_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0])

    if desc_embedding is None:
        desc_embedding = EMBEDDING_MODEL.encode(description, convert_to_tensor=True)
    if stmt_embedding is None:
        stmt_embedding = EMBEDDING_MODEL.encode(statement, convert_to_tensor=True)
    embedding_score = float(cos_sim(stmt_embedding, desc_embedding))

    return round(alpha * embedding_score + (1 - alpha) * tfidf_score, 4)

def score_chunk_level_similarity(statement, description, alpha=0.7, k=3, desc_embedding=None, sentence_embeddings=None):
    """
    Compute chunk-level hybrid similarity between a personal statement and a course description.

    The statement is split sentence by sentence. For each sentence, a weighted combination
    of TF-IDF cosine similarity and embedding cosine similarity is computed against the
    full course description. Three aggregations are returned: mean, max, and top-k mean.

    Parameters:
        statement (str): Cleaned personal statement text.
        description (str): Cleaned course description text (kept whole).
        alpha (float): Weight given to the embedding score (default 0.7).
                       TF-IDF weight = 1 - alpha.
        k (int): Number of top-scoring sentences used for the top-k mean (default 3).
        desc_embedding: Pre-computed embedding for description. Encoded on-the-fly if None.
        sentence_embeddings: Pre-computed embeddings for each sentence. Encoded on-the-fly if None.

    Returns:
        dict with keys 'avg', 'max', 'top_k', or None if inputs are invalid.
    """
    if not statement or not description:
        return None

    sentences = nltk.sent_tokenize(statement)
    if not sentences:
        return None

    if desc_embedding is None:
        desc_embedding = EMBEDDING_MODEL.encode(description, convert_to_tensor=True)
    if sentence_embeddings is None:
        sentence_embeddings = EMBEDDING_MODEL.encode(sentences, convert_to_tensor=True)

    # Fit TF-IDF once across all sentences and the description together
    tfidf_matrix = TfidfVectorizer().fit_transform(sentences + [description])
    tfidf_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:]).flatten()

    chunk_scores = [
        alpha * float(cos_sim(sent_emb, desc_embedding)) + (1 - alpha) * float(tfidf_score)
        for sent_emb, tfidf_score in zip(sentence_embeddings, tfidf_scores)
    ]

    sorted_scores = sorted(chunk_scores, reverse=True)
    top_k_scores = sorted_scores[:k]

    return {
        "avg": round(sum(chunk_scores) / len(chunk_scores), 4),
        "max": round(sorted_scores[0], 4),
        "top_k": round(sum(top_k_scores) / len(top_k_scores), 4),
    }

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
