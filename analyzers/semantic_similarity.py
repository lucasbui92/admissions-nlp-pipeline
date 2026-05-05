import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils.cleaning import clean_text_for_semantics
from utils.processing import get_optional_value
from config.schema import SEMANTIC_SOURCE_MAP

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

def precompute_statement_embeddings(df, schema):
    statements = [
        clean_text_for_semantics(row[schema["statement_col"]]) or ""
        for _, row in df.iterrows()
    ]
    return EMBEDDING_MODEL.encode(statements, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

def precompute_sentence_embeddings(df, schema):
    all_sentences = []
    sentence_counts = []

    for _, row in df.iterrows():
        cleaned = clean_text_for_semantics(row[schema["statement_col"]])
        sentences = nltk.sent_tokenize(cleaned) if cleaned else []
        all_sentences.extend(sentences)
        sentence_counts.append(len(sentences))

    if not all_sentences:
        return [None] * len(df)

    all_embeddings = EMBEDDING_MODEL.encode(all_sentences, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

    result = []
    offset = 0
    for count in sentence_counts:
        result.append(all_embeddings[offset:offset + count] if count > 0 else None)
        offset += count
    return result

def precompute_course_embeddings(course_desc_df):
    embeddings = {}
    for _, row in course_desc_df.iterrows():
        idx = int(row["index"])
        embeddings[idx] = {}
        for col in SEMANTIC_SOURCE_MAP.values():
            desc = row.get(col)
            if pd.notna(desc) and str(desc).strip():
                embeddings[idx][col] = EMBEDDING_MODEL.encode(str(desc), convert_to_tensor=True)
            else:
                embeddings[idx][col] = None
    return embeddings

def process_document_level_semantic(row, schema, data_source_type, course_desc_df=None, course_embeddings=None, stmt_embedding=None):
    """
    Compute TF-IDF cosine similarity between a student's personal statement
    and each institution's course description for their matched subject.

    Parameters:
        row: DataFrame row for one student.
        schema: Column name mapping for this data source type.
        data_source_type: "sample" or "restricted".
        course_desc_df: DataFrame with 'subject', 'description_essex',
                        'description_manchester', 'description_ucas', and
                        'combined_description' columns. If None, all scores are None.

    Returns:
        dict with identity fields and a 'doc_semantic_result' sub-dict containing
        one score per institution plus a combined score.
    """
    raw_statement = row[schema["statement_col"]]
    cleaned_statement = clean_text_for_semantics(raw_statement)

    subject_col = schema.get("subject_col")
    subject = row[subject_col] if subject_col and subject_col in row.index else None

    semantic_result = {score_key: None for score_key in SEMANTIC_SOURCE_MAP}

    if cleaned_statement and course_desc_df is not None and subject and pd.notna(subject):
        subset = course_desc_df[course_desc_df["index"] == int(subject)]
        if not subset.empty:
            desc_row = subset.iloc[0]
            precomputed = course_embeddings.get(int(subject), {}) if course_embeddings else {}
            for score_key, col in SEMANTIC_SOURCE_MAP.items():
                description = desc_row.get(col)
                if pd.notna(description) and str(description).strip():
                    semantic_result[score_key] = score_document_level_similarity(
                        cleaned_statement, str(description),
                        desc_embedding=precomputed.get(col),
                        stmt_embedding=stmt_embedding,
                    )

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": subject,
            "doc_semantic_result": semantic_result,
        }
    elif data_source_type == "restricted":
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": get_optional_value(row, schema.get("course_title")),
            "doc_semantic_result": semantic_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

def process_chunk_level_semantic(row, schema, data_source_type, course_desc_df=None, course_embeddings=None, sentence_embeddings=None):
    raw_statement = row[schema["statement_col"]]
    cleaned_statement = clean_text_for_semantics(raw_statement)

    subject_col = schema.get("subject_col")
    subject = row[subject_col] if subject_col and subject_col in row.index else None

    semantic_result = {score_key: None for score_key in SEMANTIC_SOURCE_MAP}

    if cleaned_statement and course_desc_df is not None and subject and pd.notna(subject):
        subset = course_desc_df[course_desc_df["index"] == int(subject)]
        if not subset.empty:
            desc_row = subset.iloc[0]
            precomputed = course_embeddings.get(int(subject), {}) if course_embeddings else {}
            for score_key, col in SEMANTIC_SOURCE_MAP.items():
                description = desc_row.get(col)
                if pd.notna(description) and str(description).strip():
                    semantic_result[score_key] = score_chunk_level_similarity(
                        cleaned_statement, str(description),
                        desc_embedding=precomputed.get(col),
                        sentence_embeddings=sentence_embeddings,
                    )

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": subject,
            "chunk_semantic_result": semantic_result,
        }
    elif data_source_type == "restricted":
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": get_optional_value(row, schema.get("course_title")),
            "chunk_semantic_result": semantic_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
