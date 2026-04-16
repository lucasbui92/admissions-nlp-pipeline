import pandas as pd

from utils.scoring import (
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

def add_label_lookups(course_desc_df):
    """
    Add lookup tables from course_descriptions.xlsx for label assignment.

    Returns:
        subject_set    : set of reference subjects in lowercase
        related_lookup : dict mapping related term (lowercase) → reference subject
    """
    subject_set = {
        str(s).lower()
        for s in course_desc_df["subject"].dropna()
    }

    related_lookup = {}
    if "related_terms" in course_desc_df.columns:
        for _, row in course_desc_df.iterrows():
            subject = row.get("subject")
            related_raw = row.get("related_terms")
            if pd.isna(subject) or pd.isna(related_raw):
                continue
            for term in str(related_raw).split(";"):
                term = term.strip()
                if term:
                    related_lookup[term.lower()] = str(subject)

    return subject_set, related_lookup

def assign_label(subject, course_title, subject_set, related_lookup):
    """
    Assign a match label for a single record.

    - "Exact Match"   : subject matches a reference subject exactly (case-insensitive)
    - "Related Terms" : application_course_titlemain is in the related_terms of
                        a reference subject
    - ""              : no match
    """
    subject_val = normalize_text(subject)
    title_val = normalize_text(course_title)

    if subject_val and subject_val in subject_set:
        return "Exact Match"

    if title_val and title_val in related_lookup:
        return "Related Terms"

    return ""

def add_matched_subject_column(
    student_df,
    course_desc_df,
    schema,
    student_course_col="applicationCourse_titlemain",
    subject_col="subject"
):
    subject_lookup = {}
    for subject in course_desc_df[subject_col].dropna().unique():
        subject_lookup[normalize_text(subject)] = subject

    student_df = student_df.copy()

    matched_values = []
    for value in student_df[student_course_col]:
        normalized_value = normalize_text(value)
        matched_subject = subject_lookup.get(normalized_value)
        matched_values.append(matched_subject)

    student_df[schema["subject_col"]] = matched_values
    return student_df

def process_writing_quality(row, schema, data_source_type, tool=get_language_tool(), subject_set=None, related_lookup=None):
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
        subject_value = get_optional_value(row, schema.get("subject_col"))
        label = assign_label(subject_value, course_title_value, subject_set or set(), related_lookup or {})

        grammar_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "subject": subject_value,
            "label": label,
            "grammar_result": grammar_result
        }

        readability_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "subject": subject_value,
            "label": label,
            "readability_result": readability_result
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
    return grammar_record, readability_record

def process_chunk_level_semantic(row, schema, data_source_type, course_desc_df=None, subject_set=None, related_lookup=None):
    raw_statement = row[schema["statement_col"]]
    cleaned_statement = clean_text_for_semantics(raw_statement)

    subject_col = schema.get("subject_col")
    subject = row[subject_col] if subject_col and subject_col in row.index else None

    semantic_result = {score_key: None for score_key in SEMANTIC_SOURCE_MAP}

    if cleaned_statement and course_desc_df is not None and subject and pd.notna(subject):
        subset = course_desc_df[
            course_desc_df["subject"].str.lower() == str(subject).strip().lower()
        ]
        if not subset.empty:
            desc_row = subset.iloc[0]
            for score_key, col in SEMANTIC_SOURCE_MAP.items():
                description = desc_row.get(col)
                if pd.notna(description) and str(description).strip():
                    semantic_result[score_key] = score_chunk_level_similarity(cleaned_statement, str(description))

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": subject,
            "chunk_semantic_result": semantic_result,
        }
    elif data_source_type == "restricted":
        course_title = get_optional_value(row, schema.get("course_title"))
        label = assign_label(subject, course_title, subject_set or set(), related_lookup or {})
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": course_title,
            "subject": subject,
            "label": label,
            "chunk_semantic_result": semantic_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

def process_document_level_semantic(row, schema, data_source_type, course_desc_df=None, subject_set=None, related_lookup=None):
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
        subject_set: set of reference subjects (lowercase) for label assignment.
        related_lookup: dict mapping related term (lowercase) → reference subject.

    Returns:
        dict with identity fields, a 'label' field, and a 'doc_semantic_result'
        sub-dict containing one score per institution plus a combined score.
    """
    raw_statement = row[schema["statement_col"]]
    cleaned_statement = clean_text_for_semantics(raw_statement)

    subject_col = schema.get("subject_col")
    subject = row[subject_col] if subject_col and subject_col in row.index else None

    semantic_result = {score_key: None for score_key in SEMANTIC_SOURCE_MAP}

    if cleaned_statement and course_desc_df is not None and subject and pd.notna(subject):
        subset = course_desc_df[
            course_desc_df["subject"].str.lower() == str(subject).strip().lower()
        ]
        if not subset.empty:
            desc_row = subset.iloc[0]
            for score_key, col in SEMANTIC_SOURCE_MAP.items():
                description = desc_row.get(col)
                if pd.notna(description) and str(description).strip():
                    semantic_result[score_key] = score_document_level_similarity(cleaned_statement, str(description))

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": subject,
            "doc_semantic_result": semantic_result,
        }
    elif data_source_type == "restricted":
        course_title = get_optional_value(row, schema.get("course_title"))
        label = assign_label(subject, course_title, subject_set or set(), related_lookup or {})
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": course_title,
            "subject": subject,
            "label": label,
            "doc_semantic_result": semantic_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
