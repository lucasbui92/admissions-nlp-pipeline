import language_tool_python

from utils.preprocessing import get_optional_value


def get_language_tool():
    return language_tool_python.LanguageTool("en-UK")

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

def score_grammar_quality(text, tool):
    """
    Evaluate grammatical quality using LanguageTool and compute
    a document-level grammar score.

    Returns
        dict with:
            final_score
            error_count
            word_count
            char_count
            matches
    """
    # Handle empty text
    if not text or not isinstance(text, str):
        return {
            "final_score": None,
            "error_count": 0,
            "word_count": 0,
            "char_count": 0,
            "matches": []
        }

    matches = tool.check(text)

    word_count = len(text.split())
    char_count = len(text)   # character count including spaces
    error_count = len(matches)

    # Grammar score based on word-normalised error proportion
    error_ratio = 0 if word_count == 0 else error_count / word_count
    final_score = 100 * (1 - error_ratio)

    match_list = []
    for m in matches:
        item = {
            "message": m.message,
            "context": m.context,
            "category": getattr(m.category, "id", str(m.category))
        }
        match_list.append(item)

    return {
        "final_score": round(final_score, 2),
        "error_count": error_count,
        "word_count": word_count,
        "char_count": char_count,
        "matches": match_list
    }

def process_grammar(row, schema, data_source_type, tool):
    raw_statement = row[schema["statement_col"]]
    grammar_result = score_grammar_quality(raw_statement, tool)

    if data_source_type == "sample":
        return {
            "index": row[schema["index_col"]],
            "subject": row[schema["subject_col"]],
            "grammar_result": grammar_result,
        }
    elif data_source_type == "restricted":
        return {
            "app_id": row[schema["app_id_col"]],
            "admit_year": row[schema["admit_year_col"]],
            "application_course": get_optional_value(row, schema.get("course_col")),
            "application_course_titlemain": get_optional_value(row, schema.get("course_title")),
            "grammar_result": grammar_result,
        }
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
