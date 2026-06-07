import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from config.models import EMBEDDING_MODEL
from config.settings import TOPIC_SETTINGS
from utils.cleaning import clean_text_for_semantics

nltk.download("stopwords", quiet=True)


STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(text):
    if not text:
        return text
    return " ".join(word for word in text.split() if word not in STOPWORDS)


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

def run_bertopic(docs, embeddings):
    print(
        f"Topic modelling settings:\n"
        f"  umap.n_neighbors:       {TOPIC_SETTINGS['umap']['n_neighbors']}\n"
        f"  umap.n_components:      {TOPIC_SETTINGS['umap']['n_components']}\n"
        f"  hdbscan.min_cluster_size: {TOPIC_SETTINGS['hdbscan']['min_cluster_size']}\n"
        f"  bertopic.temperature:   {TOPIC_SETTINGS['bertopic']['temperature']}"
    )

    umap_model = UMAP(
        n_neighbors=TOPIC_SETTINGS["umap"]["n_neighbors"],
        n_components=TOPIC_SETTINGS["umap"]["n_components"],
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=TOPIC_SETTINGS["hdbscan"]["min_cluster_size"],
        metric="euclidean",
        cluster_selection_method="eom",
    )
    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        min_df=TOPIC_SETTINGS["vectorizer"]["min_df"],
        max_df=TOPIC_SETTINGS["vectorizer"]["max_df"],
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=None,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    topic_ids = sorted([t for t in set(topics) if t != -1])
    topics_array = np.array(topics)
    centroids = []
    for t in topic_ids:
        centroid = embeddings[topics_array == t].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    temperature = TOPIC_SETTINGS["bertopic"]["temperature"]
    similarities = cosine_similarity(embeddings, centroids)
    scaled = similarities / temperature
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    return topics, probs, topic_ids, topic_model


def build_topic_results(topic_ids, probs, topic_model, df, schema):
    topics_section = []
    for t in topic_ids:
        keywords = []
        for word, _ in topic_model.get_topic(t):
            keywords.append(word)
        topics_section.append({"topic_number": t, "keywords": keywords})

    top_k = TOPIC_SETTINGS["output"]["top_k"]
    id_col = schema.get("app_id_col", schema.get("index_col"))
    applications_section = []
    for i, (_, row) in enumerate(df.iterrows()):
        topic_probs = {}
        for j in range(len(topic_ids)):
            if probs[i][j] > 1e-6:
                topic_probs[f"Topic {topic_ids[j]}"] = float(probs[i][j])
        top_probs = dict(sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)[:top_k])
        applications_section.append({"app_id": row[id_col], "topic_probabilities": top_probs})

    return {"topics": topics_section, "applications": applications_section}
