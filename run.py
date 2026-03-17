import argparse, json
import pandas as pd

from config.paths import resolve_paths
from config.schema import SCHEMA

from utils.processing import process_row
from utils.exporting import export_grammar_to_excel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["sample", "restricted"])
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    paths = resolve_paths(args.mode, args.input)
    schema = SCHEMA[paths.data_source_type]

    df = pd.read_excel(paths.input_file)

    grammar_results = []
    readability_results = []

    for _, row in df.iterrows():
        grammar_record, readability_record = process_row(
            row=row,
            schema=schema,
            data_source_type=paths.data_source_type,
        )

        grammar_results.append(grammar_record)
        readability_results.append(readability_record)

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.grammar_output_file, "w", encoding="utf-8") as f:
        json.dump(grammar_results, f, indent=4, ensure_ascii=False)

    with open(paths.readability_output_file, "w", encoding="utf-8") as f:
        json.dump(readability_results, f, indent=4, ensure_ascii=False)

    grammar_excel_file = export_grammar_to_excel(
        grammar_results=grammar_results,
        mode=args.mode,
    )

    print(f"Grammar output JSON → {paths.grammar_output_file}")
    print(f"Readability output JSON → {paths.readability_output_file}")
    print(f"Grammar output Excel → {grammar_excel_file}")


if __name__ == "__main__":
    main()
