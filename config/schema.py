DATA_SOURCE = {
    "sample": {
        "statement_col": "personal_statement",
        "index_col": "index",
        "subject_col": "subject"
    },
    "restricted": {
        "statement_col": "personal_statement",
        "index_col": "index",
        "subject_col": "subject"
    },
    "external_raw": {
        "statement_col": "StatementText",
        "app_id_col": "ApplicantNumber",
        "admit_year_col": "YearOfEntry"
    }
}

GRAMMAR_EXPORT_MAP = {
    "final_score": "FinalScore",
    "error_count": "ErrorCount",
    "word_count": "WordCount",
    "char_count": "CharCount",
}

READABILITY_EXPORT_MAP = {
    "flesch_reading_ease": "FleschReadingEase",
    "flesch_kincaid_grade": "FleschKincaidGrade",
    "smog_index": "SMOGIndex",
    "automated_readability_index": "AutomatedReadabilityIndex",
    "gunning_fog_index": "GunningFogIndex",
    "linsear_write_formula": "LinsearWriteFormula",
}
