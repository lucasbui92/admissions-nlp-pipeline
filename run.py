import argparse, json
import pandas as pd

from config.paths import COURSES_FILE, TOPIC_KEYWORDS_FILE, resolve_paths
from config.schema import ALL_METRICS, DATA_SOURCE

from utils.exporting import export_results_to_excel
from utils.preprocessing import (
    precompute_course_embeddings,
    precompute_statement_embeddings,
    precompute_sentence_embeddings,
)
from analyzers.semantic_similarity import (
    process_document_level_semantic,
    process_chunk_level_semantic,
)
from analyzers.grammar import get_language_tool, process_grammar
from analyzers.readability import process_readability
from analyzers.topic_modelling import find_related_keywords, load_seed_keywords, prepare_topic_docs


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
            "grammar, readability, topic_modelling. Defaults to all metrics when omitted."
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

    tool = get_language_tool() if metrics & {"grammar", "readability"} else None

    for i, (_, row) in enumerate(df.iterrows()):
        if "grammar" in metrics:
            grammar_results.append(process_grammar(row, schema, paths.data_source_type, tool))
        if "readability" in metrics:
            readability_results.append(process_readability(row, schema, paths.data_source_type))

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

    topic_docs = None
    topic_candidates = None
    if "topic_modelling" in metrics:
        topic_docs = prepare_topic_docs(df, schema)
        print(f"Topic modelling Step 1 complete: {len(topic_docs)} documents prepared.")
        seed_categories = load_seed_keywords(TOPIC_KEYWORDS_FILE)
        topic_candidates = find_related_keywords(df, schema, seed_categories)

    has_json_output = any(r is not None for r in [grammar_results, readability_results, doc_semantic_results, chunk_semantic_results])
    if has_json_output:
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

    if topic_candidates is not None:
        paths.topic_candidates_file.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for category, kw_list in topic_candidates.items():
            for rank, (keyword, freq) in enumerate(kw_list, start=1):
                rows.append({"category": category, "rank": rank, "keyword": keyword, "frequency": freq})
        pd.DataFrame(rows).to_csv(paths.topic_candidates_file, index=False)
        print(f"Topic candidates CSV → {paths.topic_candidates_file}")

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
    if excel_file is not None:
        print(f"Excel output → {excel_file}")


if __name__ == "__main__":
    main()
