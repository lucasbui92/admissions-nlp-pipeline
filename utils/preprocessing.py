import re
import nltk
import numpy as np
import pandas as pd

from config.models import EMBEDDING_MODEL
from config.schema import SEMANTIC_SOURCE_MAP
from config.settings import SEMANTIC_SETTINGS
from utils.cleaning import clean_text_for_semantics


def get_optional_value(row, col_name):
    if col_name and col_name in row.index:
        return row[col_name]
    return None

def normalize_text(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()

def precompute_statement_embeddings(df, schema):
    statements = []
    for _, row in df.iterrows():
        cleaned = clean_text_for_semantics(row[schema["statement_col"]])
        statements.append(cleaned or "")
    return EMBEDDING_MODEL.encode(statements, batch_size=SEMANTIC_SETTINGS["encoding"]["batch_size"], convert_to_tensor=True, show_progress_bar=True)

def precompute_sentence_embeddings(df, schema):
    """Unlike tokenize_statements_to_sentences, this keeps every sentence unfiltered
    and converts them straight to embedding tensors — output is vectors, not text."""
    all_sentences = []
    sentence_counts = []

    for _, row in df.iterrows():
        cleaned = clean_text_for_semantics(row[schema["statement_col"]])
        sentences = nltk.sent_tokenize(cleaned) if cleaned else []
        all_sentences.extend(sentences)
        sentence_counts.append(len(sentences))

    if not all_sentences:
        return [None] * len(df)

    all_embeddings = EMBEDDING_MODEL.encode(all_sentences, batch_size=SEMANTIC_SETTINGS["encoding"]["batch_size"], convert_to_tensor=True, show_progress_bar=True)

    result = []
    offset = 0
    for count in sentence_counts:
        result.append(all_embeddings[offset:offset + count] if count > 0 else None)
        offset += count
    return result

def tokenize_statements_to_sentences(df, schema):
    """Unlike precompute_sentence_embeddings, this returns (stmt_id, sentence) text tuples
    with noise filtered out (< 3 words, punctuation-only) — for steps that need readable sentences, not vectors."""
    id_col = schema.get("index_col") or schema.get("app_id_col")
    results = []
    processed = 0

    for _, row in df.iterrows():
        raw = row[schema["statement_col"]]
        if not isinstance(raw, str) or not raw.strip():
            continue

        stmt_id = row[id_col] if id_col and id_col in row.index else row.name
        processed += 1

        for sentence in nltk.sent_tokenize(raw):
            sentence = re.sub(r"\s+", " ", sentence).strip().lower()
            if len(sentence.split()) < 3:
                continue
            if not re.search(r"[a-z0-9]", sentence):
                continue
            results.append((stmt_id, sentence))

    return results, processed


def precompute_course_embeddings(course_desc_df):
    embeddings = {}
    for _, row in course_desc_df.iterrows():
        idx = int(row["index"])
        embeddings[idx] = {}
        for col in SEMANTIC_SOURCE_MAP.values():
            desc = row.get(col)
            if pd.notna(desc) and str(desc).strip():
                embeddings[idx][col] = EMBEDDING_MODEL.encode(str(desc), convert_to_tensor=True)
            else:
                embeddings[idx][col] = None

    return embeddings
