import argparse, json
import pandas as pd

from config.paths import COURSES_FILE, resolve_paths
from config.schema import DATA_SOURCE

from utils.exporting import export_results_to_excel
from utils.processing import (
    process_document_level_semantic,
    process_chunk_level_semantic,
    process_writing_quality,
    add_matched_subject_column,
    add_label_lookups,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["sample", "restricted"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_name", required=True, type=str)
    parser.add_argument("--include_matches", action="store_true",
            help="Include grammar match details in the Excel export."
    )
    args = parser.parse_args()
    paths = resolve_paths(args.mode, args.input, args.output_name)

    df = pd.read_excel(paths.input_file)
    if "applicationCourse_titlemain" in df.columns:
        df = add_matched_subject_column(
            df, 
            pd.read_excel(COURSES_FILE),
            DATA_SOURCE[paths.data_source_type]
        )

    course_desc_df = pd.read_excel(COURSES_FILE)
    subject_set, related_lookup = add_label_lookups(course_desc_df)

    grammar_results = []
    readability_results = []
    doc_semantic_results = []
    chunk_semantic_results = []
    for _, row in df.iterrows():
        grammar_record, readability_record = process_writing_quality(
            row,
            DATA_SOURCE[paths.data_source_type],
            paths.data_source_type,
            subject_set=subject_set,
            related_lookup=related_lookup,
        )
        grammar_results.append(grammar_record)
        readability_results.append(readability_record)

        doc_semantic_record = process_document_level_semantic(
            row,
            DATA_SOURCE[paths.data_source_type],
            paths.data_source_type,
            course_desc_df,
            subject_set=subject_set,
            related_lookup=related_lookup,
        )
        doc_semantic_results.append(doc_semantic_record)

        chunk_semantic_record = process_chunk_level_semantic(
            row,
            DATA_SOURCE[paths.data_source_type],
            paths.data_source_type,
            course_desc_df,
            subject_set=subject_set,
            related_lookup=related_lookup,
        )
        chunk_semantic_results.append(chunk_semantic_record)

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.grammar_output_file, "w", encoding="utf-8") as f:
        json.dump(grammar_results, f, indent=4, ensure_ascii=False)
    with open(paths.readability_output_file, "w", encoding="utf-8") as f:
        json.dump(readability_results, f, indent=4, ensure_ascii=False)
    with open(paths.doc_semantic_output_file, "w", encoding="utf-8") as f:
        json.dump(doc_semantic_results, f, indent=4, ensure_ascii=False)
    with open(paths.chunk_semantic_output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_semantic_results, f, indent=4, ensure_ascii=False)

    excel_file = export_results_to_excel(
        grammar_results,
        readability_results,
        doc_semantic_results,
        chunk_semantic_results,
        DATA_SOURCE[paths.data_source_type],
        paths.data_source_type,
        args.output_name,
        args.include_matches,
    )

    print(f"Grammar output JSON → {paths.grammar_output_file}")
    print(f"Readability output JSON → {paths.readability_output_file}")
    print(f"Document-level semantic output JSON → {paths.doc_semantic_output_file}")
    print(f"Chunk semantic output JSON → {paths.chunk_semantic_output_file}")
    print(f"Excel output → {excel_file}")


if __name__ == "__main__":
    main()
