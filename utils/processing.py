import nltk
import pandas as pd

from utils.scoring import (
    EMBEDDING_MODEL,
    get_language_tool,
    score_grammar_quality,
    score_readability,
    score_document_level_similarity,
    score_chunk_level_similarity,
)
from utils.cleaning import clean_text_for_semantics
from config.schema import SEMANTIC_SOURCE_MAP


def get_optional_value(row, col_name):
    if col_name and col_name in row.index:
        return row[col_name]
    return None

def normalize_text(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()

def process_writing_quality(row, schema, data_source_type, tool=get_language_tool()):
    raw_statement = row[schema["statement_col"]]

    grammar_result = score_grammar_quality(raw_statement, tool)
    readability_result = score_readability(raw_statement)

    if data_source_type == "sample":
        subject_value = row[schema["subject_col"]]

        grammar_record = {
            "index": row[schema["index_col"]],
            "subject": subject_value,
            "grammar_result": grammar_result
        }

        readability_record = {
            "index": row[schema["index_col"]],
            "subject": subject_value,
            "readability_result": readability_result
        }
    elif data_source_type == "restricted":
        course_value = get_optional_value(row, schema.get("course_col"))
        course_title_value = get_optional_value(row, schema.get("course_title"))

        grammar_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "grammar_result": grammar_result
        }

        readability_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "readability_result": readability_result
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
    return grammar_record, readability_record

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
