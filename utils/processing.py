import pandas as pd

from utils.scoring import (
    get_language_tool, 
    score_grammar_quality, 
    score_readability
)
from utils.cleaning import clean_text_for_semantics


def get_optional_value(row, col_name):
    if col_name and col_name in row.index:
        return row[col_name]
    return None

def normalize_text(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()

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

def process_writing_quality(row, schema, data_source_type, tool=get_language_tool()):
    raw_statement = row[schema["statement_col"]]

    grammar_result = score_grammar_quality(raw_statement, tool)
    readability_result = score_readability(raw_statement)

    if data_source_type == "sample":
        grammar_record = {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            "grammar_result": grammar_result
        }

        readability_record = {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            "readability_result": readability_result
        }
    elif data_source_type == "restricted":
        course_value = get_optional_value(row, schema.get("course_col"))
        course_title_value = get_optional_value(row, schema.get("course_title"))
        subject_value = get_optional_value(row, schema.get("subject_col"))

        grammar_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "subject": subject_value,
            "grammar_result": grammar_result
        }

        readability_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "application_course_titlemain": course_title_value,
            "subject": subject_value,
            "readability_result": readability_result
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

    return grammar_record, readability_record

def process_semantic_alignment(row, schema, data_source_type):
    raw_statement = row[schema["statement_col"]]

    cleaned_statement = clean_text_for_semantics(raw_statement)

    return
