import pandas as pd

from pathlib import Path
from datetime import datetime
from config.schema import GRAMMAR_EXPORT_MAP, READABILITY_EXPORT_MAP, SEMANTIC_EXPORT_MAP


EXCEL_EXPORT_DIR = Path(r"B:\derived")


def flatten_base_identifiers(record, schema, data_source_type):
    """
    Map internal record identifier keys to export column names.
    """
    if data_source_type == "restricted":
        row = {
            schema["app_id_col"]: record.get("app_id"),
            schema["admit_year_col"]: record.get("admit_year"),
        }

        course_col = schema.get("course_col")
        if course_col:
            row[course_col] = record.get("application_course")

        course_title_col = schema.get("course_title")
        if course_title_col:
            row[course_title_col] = record.get("application_course_titlemain")

        return row
    elif data_source_type == "sample":
        return {
            schema["index_col"]: record.get("index"),
            schema["subject_col"]: record.get("subject"),
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")

def flatten_grammar_record(record, schema, data_source_type, include_matches=False):
    """
    Flatten one grammar record for Excel export.
    """
    row = flatten_base_identifiers(record, schema, data_source_type)

    grammar = record.get("grammar_result", {})
    for source_key, output_key in GRAMMAR_EXPORT_MAP.items():
        row[output_key] = grammar.get(source_key)

    if include_matches:
        matches = grammar.get("matches", [])
        for i, match in enumerate(matches, start=1):
            row[f"Match{i}"] = (
                f"message: {match.get('message', '')} | "
                f"context: {match.get('context', '')} | "
                f"category: {match.get('category', '')}"
            )
    return row

def flatten_readability_record(record, schema, data_source_type):
    """
    Flatten one readability record for Excel export.
    """
    row = flatten_base_identifiers(record, schema, data_source_type)

    readability = record.get("readability_result", {})
    for source_key, output_key in READABILITY_EXPORT_MAP.items():
        row[output_key] = readability.get(source_key)
    return row


def flatten_doc_semantic_record(record, schema, data_source_type):
    """
    Flatten one document-level semantic record for Excel export.
    """
    row = flatten_base_identifiers(record, schema, data_source_type)

    semantic = record.get("doc_semantic_result", {})
    for source_key, output_key in SEMANTIC_EXPORT_MAP.items():
        row[output_key] = semantic.get(source_key)
    return row

def flatten_chunk_semantic_record(record, schema, data_source_type):
    """
    Flatten one chunk-level semantic record for Excel export.

    Each institution score is a dict with 'avg', 'max', 'top_k' keys,
    expanded into separate columns (e.g. EssexScore_Avg, EssexScore_Max, EssexScore_TopK).
    """
    row = flatten_base_identifiers(record, schema, data_source_type)

    semantic = record.get("chunk_semantic_result", {})
    for source_key, output_key in SEMANTIC_EXPORT_MAP.items():
        scores = semantic.get(source_key)
        if isinstance(scores, dict):
            row[f"{output_key}_Avg"]   = scores.get("avg")
            row[f"{output_key}_Max"]   = scores.get("max")
            row[f"{output_key}_TopK"]  = scores.get("top_k")
        else:
            row[f"{output_key}_Avg"]   = None
            row[f"{output_key}_Max"]   = None
            row[f"{output_key}_TopK"]  = None
    return row


def export_results_to_excel(
    grammar_results,
    readability_results,
    doc_semantic_results,
    chunk_semantic_results,
    schema,
    data_source_type,
    output_name,
    include_matches=False,
):
    EXCEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    output_file = EXCEL_EXPORT_DIR / f"{output_name}_{today}.xlsx"

    grammar_rows = [
        flatten_grammar_record(record, schema, data_source_type, include_matches=include_matches)
        for record in grammar_results
    ]

    readability_rows = [
        flatten_readability_record(record, schema, data_source_type)
        for record in readability_results
    ]

    doc_semantic_rows = [
        flatten_doc_semantic_record(record, schema, data_source_type)
        for record in doc_semantic_results
    ]

    chunk_semantic_rows = [
        flatten_chunk_semantic_record(record, schema, data_source_type)
        for record in chunk_semantic_results
    ]

    grammar_df = pd.DataFrame(grammar_rows)
    readability_df = pd.DataFrame(readability_rows)
    doc_semantic_df = pd.DataFrame(doc_semantic_rows)
    chunk_semantic_df = pd.DataFrame(chunk_semantic_rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        grammar_df.to_excel(writer, sheet_name="grammar", index=False)
        readability_df.to_excel(writer, sheet_name="readability", index=False)
        doc_semantic_df.to_excel(writer, sheet_name="document", index=False)
        chunk_semantic_df.to_excel(writer, sheet_name="chunk", index=False)

    return output_file
