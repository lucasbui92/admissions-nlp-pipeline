import os, json
import pandas as pd

from src.grammar_scoring import (
    get_language_tool,
    trim_text_by_words,
    score_grammar_quality,
)
from src.readability_scoring import score_readability

from config.paths import (
    PERSONAL_STATEMENT_FILE,
    GRAMMAR_OUTPUT_FILE,
    READABILITY_OUTPUT_FILE,
    OUTPUT_DIR
)


def main():
    df = pd.read_excel(PERSONAL_STATEMENT_FILE)
    tool = get_language_tool()

    grammar_results = []
    readability_results = []

    for _, row in df.iterrows():
        statement = row["personal_statement"]

        grammar_result = score_grammar_quality(statement, tool)
        readability_result = score_readability(statement)

        grammar_results.append({
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "grammar_result": grammar_result
        })

        readability_results.append({
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "readability_result": readability_result
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAMMAR_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(grammar_results, f, indent=4, ensure_ascii=False)

    with open(READABILITY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(readability_results, f, indent=4, ensure_ascii=False)

    print(f"Grammar output → {GRAMMAR_OUTPUT_FILE}")
    print(f"Readability output → {READABILITY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
