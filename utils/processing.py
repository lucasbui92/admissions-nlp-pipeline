from utils.scoring import trim_text_by_words, get_language_tool, score_grammar_quality, score_readability

def process_row(row, schema, data_source_type, tool=get_language_tool()):
    raw_statement = row[schema["statement_col"]]
    trimmed_statement = trim_text_by_words(raw_statement)

    grammar_result = score_grammar_quality(raw_statement, tool)
    readability_result = score_readability(raw_statement)

    if data_source_type in {"sample", "restricted"}:
        grammar_record = {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            # "statement": trimmed_statement,
            "grammar_result": grammar_result
        }

        readability_record = {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            # "statement": trimmed_statement,
            "readability_result": readability_result
        }

    elif data_source_type == "external_raw":
        grammar_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            # "statement": trimmed_statement,
            "grammar_result": grammar_result
        }

        readability_record = {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            # "statement": trimmed_statement,
            "readability_result": readability_result
        }

    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

    return grammar_record, readability_record
