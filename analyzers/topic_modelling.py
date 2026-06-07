import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from config.settings import TOPIC_SETTINGS
from utils.cleaning import STOPWORDS


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

    topic_ids = []
    for t in set(topics):
        if t != -1:
            topic_ids.append(t)
    topic_ids.sort()

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
