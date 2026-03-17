from pathlib import Path
import pandas as pd


EXCEL_EXPORT_DIR = Path(r"B:\raw")


def flatten_grammar_record(record):
    row = {}

    # keep metadata
    for key, value in record.items():
        if key != "grammar_result":
            row[key] = value

    grammar = record.get("grammar_result", {})

    row["final_score"] = grammar.get("final_score")
    row["error_count"] = grammar.get("error_count")
    row["word_count"] = grammar.get("word_count")
    row["error_rate"] = grammar.get("error_rate")

    matches = grammar.get("matches", [])

    for i, match in enumerate(matches, start=1):
        message = match.get("message", "")
        category = match.get("category", "")
        context = match.get("context", "")

        condensed = (
            f"[{category}] {message}\n"
            f"Context: {context}"
        )

        row[f"match_{i}"] = condensed

    return row


def export_grammar_to_excel(grammar_results, mode):
    EXCEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = EXCEL_EXPORT_DIR / f"grammar_output_{mode}.xlsx"

    flattened_rows = [flatten_grammar_record(record) for record in grammar_results]
    df = pd.DataFrame(flattened_rows)
    df.to_excel(output_file, index=False)

    return output_file
