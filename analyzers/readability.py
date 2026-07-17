import textstat

from utils.preprocessing import get_optional_value


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

def process_readability(row, schema, data_source_type):
    raw_statement = row[schema["statement_col"]]
    readability_result = score_readability(raw_statement)

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            "readability_result": readability_result,
        }
    elif data_source_type == "restricted":
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": get_optional_value(row, schema.get("course_title")),
            "readability_result": readability_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
