import nltk
import numpy as np
import pandas as pd

from config.models import EMBEDDING_MODEL
from config.schema import SEMANTIC_SOURCE_MAP
from config.settings import SEMANTIC_SETTINGS, TOPIC_SETTINGS
from utils.cleaning import clean_text_for_semantics, remove_stopwords


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

def precompute_topic_embeddings(df, schema, cache_path=None):
    if cache_path and cache_path.exists():
        meta_path = cache_path.with_suffix(".meta")
        cached_count = int(meta_path.read_text()) if meta_path.exists() else None
        if cached_count == len(df):
            print(f"Loading cached topic embeddings from {cache_path}")
            return np.load(cache_path)
        print(f"Cache row count mismatch ({cached_count} cached vs {len(df)} current) — recomputing.")

    statements = []
    for _, row in df.iterrows():
        cleaned = remove_stopwords(clean_text_for_semantics(row[schema["statement_col"]]))
        statements.append(cleaned or "")
    embeddings = EMBEDDING_MODEL.encode(
        statements,
        batch_size=TOPIC_SETTINGS["encoding"]["batch_size"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    if cache_path:
        np.save(cache_path, embeddings)
        cache_path.with_suffix(".meta").write_text(str(len(df)))
        print(f"Topic embeddings cached to {cache_path}")
    return embeddings
