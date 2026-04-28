ALL_METRICS = {"grammar", "readability", "doc_semantic", "chunk_semantic"}

DATA_SOURCE = {
    "sample": {
        "statement_col": "personal_statement",
        "index_col": "index",
        "subject_col": "subject"
    },
    "restricted": {
        "statement_col": "StatementText",
        "app_id_col": "ApplicantNumber",
        "admit_year_col": "YearOfEntry",
        "course_col": "applicationCourse",
        "course_title": "applicationCourse_titlemain",
        "subject_col": "subject"
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

SEMANTIC_SOURCE_MAP = {
    "essex_score": "description_essex",
    "manchester_score": "description_manchester",
    "ucas_score": "description_ucas",
    "combined_score": "combined_description",
}

SEMANTIC_EXPORT_MAP = {
    "essex_score": "EssexScore",
    "manchester_score": "ManchesterScore",
    "ucas_score": "UCASScore",
    "combined_score": "CombinedScore",
}
