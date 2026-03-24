from utils.scoring import trim_text_by_words, get_language_tool, score_grammar_quality, score_readability


def get_optional_value(row, col_name):
    if col_name and col_name in row.index:
        return row[col_name]
    return None

def process_row(row, schema, data_source_type, tool=get_language_tool()):
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

        grammar_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "grammar_result": grammar_result
        }

        readability_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": course_value,
            "readability_result": readability_result
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

    return grammar_record, readability_record
