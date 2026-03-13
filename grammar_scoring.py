import os, json, nltk
import language_tool_python

import pandas as pd

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

tool = language_tool_python.LanguageTool("en-US")


def trim_text_by_words(text, max_words=30):
    """
    Generate a shortened preview of a long personal statement for output storage
    and visual inspection. The full text should still be used for grammar analysis.

    Parameters
        text (str): Full personal statement text.
        max_words (int, optional): Maximum number of words to keep in the preview (default = 30).

    Returns
        str: Trimmed text preview. If the original text exceeds the word limit, an ellipsis ("...") is appended.
    """
    if not isinstance(text, str):
        return ""

    words = text.strip().split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def score_grammar_quality(text):
    """
    Evaluate the grammatical quality of a text using LanguageTool and compute
    a simple document-level grammar score based on the proportion of detected errors.

    Parameters
        text (str): Full input text (e.g., personal statement) to be analysed.

    Returns
        dict: A dictionary containing:
            - final_score : float or None
                Grammar quality score scaled to 0-100 (higher is better).
            - error_count : int
                Total number of grammar/spelling issues detected.
            - word_count : int
                Number of words in the input text.
            - error_rate : float or None
                Ratio of detected errors to total words.
            - matches : list of dict
                Simplified list of detected issues including message, context,
                and error category for inspection or downstream analysis.
    """
    # Handle empty text
    if not text or not isinstance(text, str):
        return {
            "final_score": None,
            "error_count": 0,
            "word_count": 0,
            "error_rate": None,
            "matches": []
        }

    matches = tool.check(text)

    word_count = len(text.split())
    error_count = len(matches)

    if word_count == 0:
        error_rate = 0
    else:
        error_rate = error_count / word_count

    final_score = 100 * (1 - error_rate)

    match_list = []

    for m in matches:
        item = {}
        item["message"] = m.message
        item["context"] = m.context
        item["category"] = getattr(m.category, "id", str(m.category))

        match_list.append(item)

    result = {}
    result["final_score"] = round(final_score, 2)
    result["error_count"] = error_count
    result["word_count"] = word_count
    result["error_rate"] = round(error_rate, 4)
    result["matches"] = match_list

    return result


if __name__ == "__main__":

    input_file = "personal_statements.xlsx"
    df = pd.read_excel(input_file)

    all_results = []
    for _, row in df.iterrows():
        statement = row["statement"]
        score_result = score_grammar_quality(statement)
        record = {
            "index": row["index"],
            "subject": row["subject"],
            "statement": trim_text_by_words(statement),
            "grammar_result": score_result
        }

        all_results.append(record)

    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # Save JSON properly formatted
    with open("output/scores.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("Finished. Output saved.")
