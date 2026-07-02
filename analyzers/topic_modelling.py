import nltk
from collections import Counter

from config.models import LEMMATIZER, NLP, STOPWORDS
from config.settings import TOPIC_SETTINGS

from utils.cleaning import clean_text_for_topics

nltk.download("punkt_tab", quiet=True)


def prepare_topic_docs(df, schema):
    docs = []
    for _, row in df.iterrows():
        cleaned = clean_text_for_topics(row[schema["statement_col"]])
        docs.append(cleaned or "")
    return docs


def load_seed_keywords(filepath):
    """Parse topics_keywords.txt → {category: [cleaned_keyword, ...]}"""
    categories = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            category, kw_part = line.split(":", 1)
            category = category.strip()
            raw_keywords = [kw.strip() for kw in kw_part.split(",") if kw.strip()]
            cleaned = [clean_text_for_topics(kw) for kw in raw_keywords]
            categories[category] = [kw for kw in cleaned if kw]
    return categories


def consolidate_candidates(candidates, top_n):
    normalized = Counter()
    for phrase, count in candidates:
        normalized[phrase.replace("-", " ")] += count

    lemma_groups = {}
    for phrase in normalized:
        key = " ".join(LEMMATIZER.lemmatize(w) for w in phrase.split())
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


def find_related_keywords(df, schema, seed_categories, top_n=30):
    """
    For each category, find sentences containing a seed keyword, extract the
    other words from those sentences, lemmatize them, and return top_n by
    frequency — excluding seed words and stopwords.
    """
    all_seed_phrases = {kw for kws in seed_categories.values() for kw in kws}
    type1_generic_keywords = set(TOPIC_SETTINGS["type1_generic_keywords"])
    type1_generic_keywords |= set(TOPIC_SETTINGS.get("type1_cross_topic_generics", []))
    type2_selective_removing = {
        cat: set(words)
        for cat, words in TOPIC_SETTINGS.get("type2_selective_removing", {}).items()
    }
    category_counters = {cat: Counter() for cat in seed_categories}

    for _, row in df.iterrows():
        raw_text = row[schema["statement_col"]]
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        for sentence in nltk.sent_tokenize(raw_text):
            if not (cleaned := clean_text_for_topics(sentence)):
                continue

            sentence_words = set(cleaned.split())
            matched_categories = []
            for category, keywords in seed_categories.items():
                for kw in keywords:
                    if kw in sentence_words if " " not in kw else kw in cleaned:
                        matched_categories.append(category)
                        break

            if not matched_categories:
                continue

            doc = NLP(sentence)

            phrases = set()
            for span in list(doc.noun_chunks) + list(doc.ents):
                tokens = [t for t in span if t.pos_ not in {"DET", "PRON"}]
                for length in range(2, len(tokens) + 1):
                    for start in range(len(tokens) - length + 1):
                        phrase = " ".join(t.text.lower() for t in tokens[start:start + length])
                        phrases.add(phrase)

            for category in matched_categories:
                cat_stopwords = type2_selective_removing.get(category, set())

                for token in doc:
                    word = token.text.lower()
                    if not token.is_alpha or len(word) < 3 or word in STOPWORDS:
                        continue
                    lemma = LEMMATIZER.lemmatize(word, pos="v")
                    if lemma == word:
                        lemma = LEMMATIZER.lemmatize(word)
                    if lemma in all_seed_phrases or lemma in type1_generic_keywords:
                        continue
                    if lemma in cat_stopwords:
                        continue
                    category_counters[category][lemma] += 1

                for phrase in phrases:
                    if phrase in all_seed_phrases or phrase in type1_generic_keywords:
                        continue
                    if phrase in cat_stopwords:
                        continue
                    category_counters[category][phrase] += 1

    buffer_multiplier = TOPIC_SETTINGS["buffer_multiplier"]
    buffer_n = int(top_n * buffer_multiplier)

    results = {}
    for cat, counter in category_counters.items():
        total = len(counter)
        candidates = counter.most_common(buffer_n)
        seed_keywords = seed_categories.get(cat, [])
        seed_words = {w for seed in seed_keywords for w in seed.split()}
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
        results[cat] = consolidated
        print(f"  [{cat}] total: {total} → extracted: {len(candidates)} → seed filter: {len(filtered)} → consolidated: {len(consolidated)} (target: {top_n})")

    return results
