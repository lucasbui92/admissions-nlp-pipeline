import nltk
import torch
import yaml
from collections import Counter
from pathlib import Path

from sentence_transformers import util as st_util

from config.models import EMBEDDING_MODEL, LEMMATIZER, NLP, STOPWORDS
from config.paths import EXCLUDED_WORDS_FILE

from utils.cleaning import clean_text_for_topics

nltk.download("punkt_tab", quiet=True)

with open(Path("config") / "settings.yml") as f:
    TOPIC_SETTINGS = yaml.safe_load(f)["topic_modelling"]


def lemmatize(word):
    """Lemmatize as a verb first (catches more inflections), falling back to the default POS if unchanged."""
    lemma = LEMMATIZER.lemmatize(word, pos="v")
    if lemma == word:
        lemma = LEMMATIZER.lemmatize(word)
    return lemma


def build_topic_lookup(topics):
    """Lemmatize all keywords once upfront to avoid repeated lemmatization per sentence."""
    lookup = {}
    for topic, keywords in topics.items():
        single_words = set()
        phrases = []
        for kw in keywords:
            parts = []
            for w in kw.split():
                parts.append(lemmatize(w))
            lemmatized = " ".join(parts)
            if " " in lemmatized:
                phrases.append(lemmatized)
            else:
                single_words.add(lemmatized)
        lookup[topic] = (single_words, phrases)
    return lookup


def match_sentences_to_topics(sentences, topics):
    """
    Score each sentence against each topic by counting lemmatized keyword matches.
    Single-word keywords matched against a word set; multi-word keywords matched as substrings.
    Assigns the topic with the highest score; ties go to whichever topic appears first.
    Sentences with zero matches across all topics are assigned None.
    Returns list of (stmt_id, sentence, scores_dict, assigned_topics).
    """
    lookup = build_topic_lookup(topics)
    topic_names = list(topics.keys())
    results = []

    for i, (stmt_id, sentence) in enumerate(sentences):
        lemma_words = set()
        lemma_parts = []
        for w in sentence.split():
            if not w.isalpha():
                lemma_parts.append(w)
                continue
            lemma = lemmatize(w)
            lemma_words.add(lemma)
            lemma_parts.append(lemma)
        lemma_text = " ".join(lemma_parts)

        scores = {}
        for topic, (single_words, phrases) in lookup.items():
            score = len(lemma_words & single_words)
            score += sum(1 for phrase in phrases if phrase in lemma_text)
            scores[topic] = score

        best_score = max(scores.values())
        assigned = []
        if best_score > 0:
            for t in topic_names:
                if scores[t] == best_score:
                    assigned.append(t)
        results.append((stmt_id, sentence, scores, assigned))

        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:,} / {len(sentences):,} sentences processed...")

    return results


def build_keyword_embeddings(topics):
    """Embed all keywords upfront, stacked per topic for efficient cosine similarity in Phase 2."""
    all_pairs = []
    for topic, kws in topics.items():
        for kw in kws:
            all_pairs.append((topic, kw))

    all_texts = []
    for _, kw in all_pairs:
        all_texts.append(kw)

    all_embeddings = EMBEDDING_MODEL.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)

    topic_embeddings = {topic: [] for topic in topics}
    for (topic, _), emb in zip(all_pairs, all_embeddings):
        topic_embeddings[topic].append(emb)

    stacked_embeddings = {}
    for topic, embs in topic_embeddings.items():
        stacked_embeddings[topic] = torch.stack(embs)
    return stacked_embeddings


def apply_semantic_fallback(results, keyword_embeddings, score_threshold=2, similarity_threshold=0.75, fractional_score=0.5):
    """
    Phase 2 fallback for sentences that scored below score_threshold in Phase 1.
    Generates sentence embeddings in batch, then adds fractional_score to any topic
    whose keyword embeddings exceed similarity_threshold in cosine similarity.
    Re-assigns topic based on combined Phase 1 + Phase 2 scores.
    """
    topic_names = list(keyword_embeddings.keys())

    low_score_indices = []
    for i, (_, _, scores, _) in enumerate(results):
        if max(scores.values()) < score_threshold:
            low_score_indices.append(i)

    if not low_score_indices:
        print("  Phase 2: no sentences below threshold, skipping.")
        return results

    print(f"  Phase 2: {len(low_score_indices):,} sentences below threshold, running semantic fallback...")

    low_score_sentences = [results[i][1] for i in low_score_indices]
    sentence_embeddings = EMBEDDING_MODEL.encode(
        low_score_sentences, batch_size=256, convert_to_tensor=True, show_progress_bar=True
    )

    updated_results = list(results)
    for result_i, sent_emb in zip(low_score_indices, sentence_embeddings):
        stmt_id, sentence, scores, _ = updated_results[result_i]

        combined = dict(scores)
        for topic, topic_embs in keyword_embeddings.items():
            max_sim = st_util.cos_sim(sent_emb.unsqueeze(0), topic_embs).max().item()
            if max_sim > similarity_threshold:
                combined[topic] += fractional_score

        best_score = max(combined.values())
        assigned = []
        if best_score > 0:
            for t in topic_names:
                if combined[t] == best_score:
                    assigned.append(t)
        updated_results[result_i] = (stmt_id, sentence, combined, assigned)

    return updated_results


def aggregate_statement_topics(results, topic_names):
    """
    For each statement, count how many sentences were assigned to each topic (a sentence
    tied between multiple topics splits its count evenly across them — fair division),
    then divide by the statement's total sentence count to get proportions.
    Returns list of {"statement_id", "topic_proportions": {topic: proportion, ...}}
    covering all topic_names (0.0 for topics with no assigned sentences).
    """
    stmt_counts = {}
    stmt_sentence_totals = Counter()

    for stmt_id, _, _, assigned in results:
        stmt_sentence_totals[stmt_id] += 1
        counts = stmt_counts.setdefault(stmt_id, Counter())
        if assigned:
            share = 1 / len(assigned)
            for topic in assigned:
                counts[topic] += share

    aggregated = []
    for stmt_id, total_sentences in stmt_sentence_totals.items():
        counts = stmt_counts[stmt_id]
        proportions = {}
        for topic in topic_names:
            proportions[topic] = round(counts.get(topic, 0.0) / total_sentences, 2)
        aggregated.append({"statement_id": stmt_id, "topic_proportions": proportions})
    return aggregated


def load_excluded_words(filepath):
    """Parse excluded_words.txt into type1 generic set, type1 cross-topic set, and type2 per-topic dict."""
    type1_generic = set()
    type1_cross_topic = set()
    type2_selective = {}
    current_section = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "[TYPE1_GENERIC]":
                current_section = "type1_generic"
            elif line == "[TYPE1_CROSS_TOPIC_GENERICS]":
                current_section = "type1_cross_topic"
            elif line == "[TYPE2_SELECTIVE]":
                current_section = "type2_selective"
            elif current_section in ("type1_generic", "type1_cross_topic"):
                for word in line.split(","):
                    word = word.strip()
                    if word:
                        if current_section == "type1_generic":
                            type1_generic.add(word)
                        else:
                            type1_cross_topic.add(word)
            elif current_section == "type2_selective" and ":" in line:
                topic, _, words_part = line.partition(":")
                topic = topic.strip()
                words = set()
                for word in words_part.split(","):
                    word = word.strip()
                    if word:
                        words.add(word)
                type2_selective[topic] = words

    return type1_generic, type1_cross_topic, type2_selective


def load_seed_keywords(filepath):
    """Parse topics_keywords_seed.txt → {topic: [cleaned_keyword, ...]}"""
    topics = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            topic, kw_part = line.split(":", 1)
            topic = topic.strip()

            raw_keywords = []
            for kw in kw_part.split(","):
                kw = kw.strip()
                if kw:
                    raw_keywords.append(kw)

            cleaned = []
            for kw in raw_keywords:
                cleaned.append(clean_text_for_topics(kw))

            topics[topic] = [kw for kw in cleaned if kw]
    return topics


def consolidate_candidates(candidates, top_n):
    normalized = Counter()
    for phrase, count in candidates:
        normalized[phrase.replace("-", " ")] += count

    lemma_groups = {}
    for phrase in normalized:
        lemmas = []
        for w in phrase.split():
            lemmas.append(LEMMATIZER.lemmatize(w))
        key = " ".join(lemmas)
        lemma_groups.setdefault(key, []).append(phrase)

    merged = Counter()
    for key, group in lemma_groups.items():
        canonical = max(group, key=lambda p: normalized[p])
        merged[canonical] = sum(normalized[p] for p in group)

    phrases = set(merged.keys())
    substrings = set()
    for phrase in phrases:
        for other in phrases:
            if phrase != other and phrase in other:
                substrings.add(phrase)
                break

    for phrase in substrings:
        del merged[phrase]

    return merged.most_common(top_n)


def find_related_keywords(df, schema, seed_topics, top_n=30):
    """
    For each topic, find sentences containing a seed keyword, extract the
    other words from those sentences, lemmatize them, and return top_n by
    frequency — excluding seed words and stopwords.
    """
    all_seed_phrases = set()
    for kws in seed_topics.values():
        for kw in kws:
            all_seed_phrases.add(kw)

    type1_generic, type1_cross_topic, type2_selective_removing = load_excluded_words(EXCLUDED_WORDS_FILE)
    type1_generic_keywords = type1_generic | type1_cross_topic

    topic_counters = {}
    for topic in seed_topics:
        topic_counters[topic] = Counter()

    for _, row in df.iterrows():
        raw_text = row[schema["statement_col"]]
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        for sentence in nltk.sent_tokenize(raw_text):
            if not (cleaned := clean_text_for_topics(sentence)):
                continue

            sentence_words = set(cleaned.split())
            matched_topics = []
            for topic, keywords in seed_topics.items():
                for kw in keywords:
                    if " " not in kw:
                        kw_matches = kw in sentence_words
                    else:
                        kw_matches = kw in cleaned
                    if kw_matches:
                        matched_topics.append(topic)
                        break

            if not matched_topics:
                continue

            doc = NLP(sentence)

            phrases = set()
            for span in list(doc.noun_chunks) + list(doc.ents):
                tokens = [t for t in span if t.pos_ not in {"DET", "PRON"}]
                for length in range(2, len(tokens) + 1):
                    for start in range(len(tokens) - length + 1):
                        words = []
                        for t in tokens[start:start + length]:
                            words.append(t.text.lower())
                        phrase = " ".join(words)
                        phrases.add(phrase)

            for topic in matched_topics:
                topic_stopwords = type2_selective_removing.get(topic, set())

                for token in doc:
                    word = token.text.lower()
                    if not token.is_alpha or len(word) < 3 or word in STOPWORDS:
                        continue
                    lemma = lemmatize(word)
                    if lemma in all_seed_phrases or lemma in type1_generic_keywords:
                        continue
                    if lemma in topic_stopwords:
                        continue
                    topic_counters[topic][lemma] += 1

                for phrase in phrases:
                    if phrase in all_seed_phrases or phrase in type1_generic_keywords:
                        continue
                    if phrase in topic_stopwords:
                        continue
                    topic_counters[topic][phrase] += 1

    buffer_multiplier = TOPIC_SETTINGS["buffer_multiplier"]
    buffer_n = int(top_n * buffer_multiplier)

    results = {}
    for topic, counter in topic_counters.items():
        total = len(counter)
        candidates = counter.most_common(buffer_n)
        seed_keywords = seed_topics.get(topic, [])
        seed_words = set()
        for seed in seed_keywords:
            for w in seed.split():
                seed_words.add(w)
        filtered = []
        for phrase, count in candidates:
            is_seed_related = False
            for seed in seed_keywords:
                if phrase in seed or seed in phrase:
                    is_seed_related = True
                    break
            if not is_seed_related:
                phrase_words = set(phrase.split())
                if phrase_words & seed_words:
                    is_seed_related = True
            if not is_seed_related:
                filtered.append((phrase, count))
        consolidated = consolidate_candidates(filtered, top_n)
        results[topic] = consolidated
        print(f"  [{topic}] total: {total} → extracted: {len(candidates)} → seed filter: {len(filtered)} → consolidated: {len(consolidated)} (target: {top_n})")

    return results
