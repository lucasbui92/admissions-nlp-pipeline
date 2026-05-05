import argparse, json
import pandas as pd

from config.paths import COURSES_FILE, resolve_paths
from config.schema import ALL_METRICS, DATA_SOURCE

from utils.exporting import export_results_to_excel
from utils.processing import process_writing_quality
from analyzers.semantic_similarity import (
    process_document_level_semantic,
    process_chunk_level_semantic,
    precompute_course_embeddings,
    precompute_statement_embeddings,
    precompute_sentence_embeddings,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["sample", "restricted"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_name", required=True, type=str)
    parser.add_argument("--include_matches", action="store_true",
            help="Include grammar match details in the Excel export."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(ALL_METRICS),
        default=None,
        help=(
            "Single metric to compute. Choices: chunk_semantic, doc_semantic, "
            "grammar, readability. Defaults to all metrics when omitted."
        ),
    )
    args = parser.parse_args()
    paths = resolve_paths(args.mode, args.input, args.output_name)

    metrics = {args.metric} if args.metric else ALL_METRICS

    df = pd.read_excel(paths.input_file)
    course_desc_df = (
        pd.read_excel(COURSES_FILE)
        if metrics & {"doc_semantic", "chunk_semantic"}
        else None
    )
    course_embeddings = precompute_course_embeddings(course_desc_df) if course_desc_df is not None else None

    schema = DATA_SOURCE[paths.data_source_type]

    stmt_embeddings = precompute_statement_embeddings(df, schema) if "doc_semantic" in metrics else None
    sent_embeddings = precompute_sentence_embeddings(df, schema) if "chunk_semantic" in metrics else None

    grammar_results = [] if "grammar" in metrics else None
    readability_results = [] if "readability" in metrics else None
    doc_semantic_results = [] if "doc_semantic" in metrics else None
    chunk_semantic_results = [] if "chunk_semantic" in metrics else None

    for i, (_, row) in enumerate(df.iterrows()):
        if metrics & {"grammar", "readability"}:
            grammar_record, readability_record = process_writing_quality(
                row, schema, paths.data_source_type,
            )
            if grammar_results is not None:
                grammar_results.append(grammar_record)
            if readability_results is not None:
                readability_results.append(readability_record)

        if "doc_semantic" in metrics:
            doc_semantic_results.append(process_document_level_semantic(
                row, schema, paths.data_source_type, course_desc_df, course_embeddings,
                stmt_embedding=stmt_embeddings[i],
            ))

        if "chunk_semantic" in metrics:
            chunk_semantic_results.append(process_chunk_level_semantic(
                row, schema, paths.data_source_type, course_desc_df, course_embeddings,
                sentence_embeddings=sent_embeddings[i],
            ))

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if grammar_results is not None:
        with open(paths.grammar_output_file, "w", encoding="utf-8") as f:
            json.dump(grammar_results, f, indent=4, ensure_ascii=False)
        print(f"Grammar output JSON → {paths.grammar_output_file}")

    if readability_results is not None:
        with open(paths.readability_output_file, "w", encoding="utf-8") as f:
            json.dump(readability_results, f, indent=4, ensure_ascii=False)
        print(f"Readability output JSON → {paths.readability_output_file}")

    if doc_semantic_results is not None:
        with open(paths.doc_semantic_output_file, "w", encoding="utf-8") as f:
            json.dump(doc_semantic_results, f, indent=4, ensure_ascii=False)
        print(f"Document-level semantic output JSON → {paths.doc_semantic_output_file}")

    if chunk_semantic_results is not None:
        with open(paths.chunk_semantic_output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_semantic_results, f, indent=4, ensure_ascii=False)
        print(f"Chunk semantic output JSON → {paths.chunk_semantic_output_file}")

    excel_file = export_results_to_excel(
        grammar_results,
        readability_results,
        doc_semantic_results,
        chunk_semantic_results,
        schema,
        paths.data_source_type,
        args.output_name,
        args.include_matches,
    )
    print(f"Excel output → {excel_file}")


if __name__ == "__main__":
    main()
