import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from config.models import EMBEDDING_MODEL
from utils.cleaning import clean_text_for_semantics

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(text):
    if not text:
        return text
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def precompute_topic_embeddings(df, schema):
    statements = [
        remove_stopwords(clean_text_for_semantics(row[schema["statement_col"]])) or ""
        for _, row in df.iterrows()
    ]
    return EMBEDDING_MODEL.encode(
        statements,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

def run_bertopic(docs, embeddings, min_cluster_size=150):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        min_df=2,
        max_df=0.5,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topics, probs, topic_model


def reduce_bertopic_topics(topic_model, docs, nr_topics):
    topics, probs = topic_model.reduce_topics(docs, nr_topics=nr_topics)
    return topics, probs, topic_model


def build_topic_results(topics, probs, topic_model, df, schema, top_n=5):
    topic_ids = sorted([t for t in set(topics) if t != -1])

    topics_section = [
        {
            "topic_number": t,
            "keywords": [word for word, _ in topic_model.get_topic(t)],
        }
        for t in topic_ids
    ]

    probs_array = np.array(probs)
    if probs_array.ndim == 1:
        # BERTopic returned one scalar per doc (assigned-topic probability only);
        # reconstruct a full matrix with zeros for all other topics.
        topic_id_to_idx = {t: j for j, t in enumerate(topic_ids)}
        probs_2d = np.zeros((len(topics), len(topic_ids)))
        for i, (t, p) in enumerate(zip(topics, probs_array)):
            if t in topic_id_to_idx:
                probs_2d[i, topic_id_to_idx[t]] = float(p)
    else:
        probs_2d = probs_array

    id_col = schema.get("app_id_col", schema.get("index_col"))
    applications_section = [
        {
            "app_id": row[id_col],
            "topic_probabilities": dict(
                sorted(
                    {
                        f"Topic {topic_ids[j]}": float(probs_2d[i][j])
                        for j in range(len(topic_ids))
                        if probs_2d[i][j] > 1e-6
                    }.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_n]
            ),
        }
        for i, (_, row) in enumerate(df.iterrows())
    ]

    return {"topics": topics_section, "applications": applications_section}
