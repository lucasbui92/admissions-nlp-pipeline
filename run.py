import os, json
import pandas as pd

from src.grammar_scoring import (
    get_language_tool,
    trim_text_by_words,
    score_grammar_quality,
)
from src.readability_scoring import score_readability


def main():
    input_file = "data/sample/sample_personal_statements.xlsx"
    grammar_output_file = "output/sample/grammar.json"
    readability_output_file = "output/sample/readability.json"

    df = pd.read_excel(input_file)
    tool = get_language_tool()

    grammar_results = []
    readability_results = []

    for _, row in df.iterrows():
        statement = row["personal_statement"]

        grammar_result = score_grammar_quality(statement, tool)
        readability_result = score_readability(statement)

        grammar_record = {
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "grammar_result": grammar_result
        }

        readability_record = {
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "readability_result": readability_result
        }

        grammar_results.append(grammar_record)
        readability_results.append(readability_record)

    os.makedirs("output", exist_ok=True)

    with open(grammar_output_file, "w", encoding="utf-8") as f:
        json.dump(grammar_results, f, indent=4, ensure_ascii=False)

    with open(readability_output_file, "w", encoding="utf-8") as f:
        json.dump(readability_results, f, indent=4, ensure_ascii=False)

    print(f"Finished. Grammar output saved to {grammar_output_file}")
    print(f"Finished. Readability output saved to {readability_output_file}")


if __name__ == "__main__":
    main()
