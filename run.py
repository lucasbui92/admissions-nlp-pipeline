import os
import json
import pandas as pd

from src.grammar_scoring import get_language_tool, trim_text_by_words, score_grammar_quality


def main():
    input_file = "sample_data/sample_personal_statements.xlsx"
    output_file = "output/scores.json"

    df = pd.read_excel(input_file)
    tool = get_language_tool()

    all_results = []

    for _, row in df.iterrows():
        statement = row["personal_statement"]
        score_result = score_grammar_quality(statement, tool)

        record = {
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "grammar_result": score_result
        }

        all_results.append(record)

    os.makedirs("output", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Finished. Output saved to {output_file}")


if __name__ == "__main__":
    main()
