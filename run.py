import argparse, gzip, json, pickle, sys

import pandas as pd

from config.paths import COURSES_FILE, TOPIC_KEYWORDS_FILE, TOPIC_KEYWORDS_FINAL_FILE, resolve_paths
from config.schema import ALL_METRICS, DATA_SOURCE

from utils.exporting import export_results_to_excel
from utils.preprocessing import (
    get_optional_value,
    precompute_course_embeddings,
    precompute_statement_embeddings,
    precompute_sentence_embeddings,
    tokenize_statements_to_sentences,
)
from analyzers.semantic_similarity import (
    process_document_level_semantic,
    process_chunk_level_semantic,
)
from analyzers.grammar import get_language_tool, process_grammar
from analyzers.readability import process_readability
from analyzers.topic_modelling import (
    aggregate_statement_topics,
    apply_semantic_fallback,
    build_keyword_embeddings,
    find_related_keywords,
    load_seed_keywords,
    match_sentences_to_topics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["sample", "restricted"])
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
    parser.add_argument(
        "--stage",
        choices=["candidates", "scoring"],
        default=None,
        help=(
            "Topic modelling stage. Required when --metric topic_modelling is used. "
            "'candidates' extracts keyword candidates for review. "
            "'scoring' runs sentence matching (requires topics_keywords_final.txt)."
        ),
    )
    parser.add_argument(
        "--export_unassigned",
        action="store_true",
        help=(
            "During --stage scoring, also export a CSV of sentences that matched no "
            "topic at all. Off by default — meant as a one-time diagnostic, not for "
            "every run."
        ),
    )
    args = parser.parse_args()
    paths = resolve_paths(args.mode, args.output_name)

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

    topic_candidates = None
    if "topic_modelling" in metrics:
        if args.stage is None:
            print("ERROR: --stage is required when running topic_modelling. Choose 'candidates' or 'scoring'.")
            sys.exit(1)

        if args.export_unassigned and args.stage != "scoring":
            print("ERROR: --export_unassigned is only valid with --stage scoring.")
            sys.exit(1)

        if args.stage == "candidates":
            print("Running topic modelling candidates stage.")
            seed_topics = load_seed_keywords(TOPIC_KEYWORDS_FILE)
            topic_candidates = find_related_keywords(df, schema, seed_topics)
            print("✓ Complete. Review topic_candidates file, finalise topics_keywords_final.txt, then run --stage scoring.")

        elif args.stage == "scoring":
            if not TOPIC_KEYWORDS_FINAL_FILE.exists():
                print(f"ERROR: Approved keywords not found at {TOPIC_KEYWORDS_FINAL_FILE}")
                print("Run --stage candidates first, finalise keywords, then re-run with --stage scoring.")
                sys.exit(1)

            # Step 1: Tokenize statements into sentences
            if paths.sentences_tokenized_pkl.exists():
                with gzip.open(paths.sentences_tokenized_pkl, "rb") as f:
                    sentences = pickle.load(f)
                print(f"Step 1: cache hit — {len(sentences):,} sentences loaded from {paths.sentences_tokenized_pkl}")
            else:
                sentences, stmt_count = tokenize_statements_to_sentences(df, schema)
                paths.sentences_tokenized_pkl.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with gzip.open(paths.sentences_tokenized_pkl, "wb") as f:
                        pickle.dump(sentences, f)
                except Exception:
                    paths.sentences_tokenized_pkl.unlink(missing_ok=True)
                    raise
                print(f"Step 1 complete: {stmt_count} statements, {len(sentences):,} sentences → {paths.sentences_tokenized_pkl}")

            # Step 2: Phase 1 — lemmatization-based keyword matching
            topics = load_seed_keywords(TOPIC_KEYWORDS_FINAL_FILE)
            print(f"Step 2: matching {len(sentences):,} sentences across {len(topics)} topics...")
            results = match_sentences_to_topics(sentences, topics)
            print("Step 2 complete.")

            # Step 3: Phase 2 — semantic similarity fallback for low-scoring sentences
            print("Step 3: building keyword embeddings for semantic fallback...")
            keyword_embeddings = build_keyword_embeddings(topics)
            results = apply_semantic_fallback(results, keyword_embeddings)
            print("Step 3 complete.")

            if paths.data_source_type == "restricted":
                id_col = schema["app_id_col"]
                identifier_lookup = df.set_index(id_col)
            else:
                identifier_lookup = None

            def build_identifier_row(stmt_id):
                if identifier_lookup is not None and stmt_id in identifier_lookup.index:
                    id_row = identifier_lookup.loc[stmt_id]
                    return {
                        "ApplicantNumber": stmt_id,
                        "YearOfEntry": id_row[schema["admit_year_col"]],
                        "applicationCourse": get_optional_value(id_row, schema.get("course_col")),
                        "applicationCourse_titlemain": get_optional_value(id_row, schema.get("course_title")),
                    }
                return {"statement_id": stmt_id}

            # Step 4a: Collect sentences that matched no topic at all, before results is discarded.
            # Opt-in only (--export_unassigned) — a one-time diagnostic, not needed on every run.
            if args.export_unassigned:
                unassigned_rows = []
                last_stmt_id = None
                for stmt_id, sentence, _, assigned in results:
                    if not assigned:
                        if stmt_id == last_stmt_id:
                            id_row = {}
                            for key in unassigned_rows[-1]:
                                if key != "sentence":
                                    id_row[key] = ""
                        else:
                            id_row = build_identifier_row(stmt_id)
                            last_stmt_id = stmt_id
                        id_row["sentence"] = sentence
                        unassigned_rows.append(id_row)

                paths.topic_unassigned_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(unassigned_rows).to_csv(paths.topic_unassigned_csv, index=False)
                print(f"✓ {len(unassigned_rows):,} unassigned sentences → {paths.topic_unassigned_csv}")

            # Step 4b: Aggregate per-sentence scores into per-topic proportions per statement
            statement_topics = aggregate_statement_topics(results, topics.keys())
            del results

            rows = []
            for entry in statement_topics:
                row = build_identifier_row(entry["statement_id"])
                row.update(entry["topic_proportions"])
                rows.append(row)

            paths.topic_scoring_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(paths.topic_scoring_csv, index=False)
            print(f"✓ Complete. {len(rows):,} statements → {paths.topic_scoring_csv}")

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
        for topic, kw_list in topic_candidates.items():
            for rank, (keyword, freq) in enumerate(kw_list, start=1):
                rows.append({"topic": topic, "rank": rank, "keyword": keyword, "frequency": freq})
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
        paths.derived_dir,
        args.include_matches,
    )
    if excel_file is not None:
        print(f"Excel output → {excel_file}")


if __name__ == "__main__":
    main()
