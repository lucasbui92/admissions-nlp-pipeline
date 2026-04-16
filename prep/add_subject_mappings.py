"""
Adds and populates the related_terms column in course_descriptions.xlsx by combining:
  1. Fuzzy-matched course titles from applicationCourse_titlemain (output JSONs)
  2. Manually curated terms from config/subject_mappings.py

config/subject_mappings.py is seeded with all subjects (empty lists) on the
first run. After that, add rare terms that fuzzy matching cannot detect (e.g.
"Bookkeeping" for Accounting) directly to that file, then re-run this script.
The script will never remove or overwrite existing entries — it only adds
subjects that are newly seen.

Usage:
    python prep/add_subject_mappings.py
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import COURSES_FILE

MAPPINGS_FILE = Path("config/subject_mappings.py")

# Words too generic to count as a meaningful shared keyword between a subject
# and a course title. "science" and "sciences" appear in dozens of unrelated
# courses (Computer Science, Biomedical Science, etc.) so they are excluded
# from the overlap check to prevent cross-domain false positives.
STOPWORDS = {
    "and", "or", "with", "of", "in", "the", "a", "an",
    "science", "sciences", "studies",
}


def gather_course_titles():
    """Collect all unique applicationCourse_titlemain values from output JSONs."""
    titles = set()
    for path in glob.glob("output/**/*.json", recursive=True):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    title = record.get("application_course_titlemain")
                    if title:
                        titles.add(title)
    return sorted(titles)


def load_subject_mappings():
    """
    Load SUBJECT_MAPPINGS from config/subject_mappings.py.
    Returns an empty dict if the file is absent or the dict is empty.
    """
    if not MAPPINGS_FILE.exists():
        return {}
    namespace = {}
    exec(MAPPINGS_FILE.read_text(encoding="utf-8"), namespace)
    return namespace.get("SUBJECT_MAPPINGS", {})


def seed_subject_mappings(subjects, existing_mappings):
    """
    Ensure every subject has an entry in config/subject_mappings.py.
    Only adds subjects that are not already present — never removes or
    modifies existing entries (including any manually added terms).
    """
    new_subjects = [s for s in subjects if s not in existing_mappings]
    if not new_subjects:
        return

    updated = dict(existing_mappings)
    for subject in new_subjects:
        updated[subject] = []

    lines = ["SUBJECT_MAPPINGS = {"]
    for subject in sorted(updated):
        terms = updated[subject]
        lines.append(f"    {subject!r}: [")
        for term in terms:
            lines.append(f"        {term!r},")
        lines.append("    ],")
    lines.append("}")
    MAPPINGS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Seeded {len(new_subjects)} new subject(s) into {MAPPINGS_FILE}")


def meaningful_words(text):
    """Return the set of significant words in a string, excluding stopwords."""
    return {
        w for w in text.lower().replace(",", " ").split()
        if w not in STOPWORDS and len(w) > 2
    }


def shares_keyword(subject, course_title):
    """True if subject and course title share at least one meaningful word."""
    return bool(meaningful_words(subject) & meaningful_words(course_title))


def add_related_terms(subjects, course_titles, manual_mappings):
    """
    For each subject, produce a merged list of related terms from:
      - keyword matching against course_titles (any course sharing a meaningful
        word with the subject is included)
      - manually curated terms in manual_mappings

    Rules:
    - Exact match (case-insensitive) between course title and subject is skipped.
    - A course is included if it shares at least one meaningful word with the
      subject (excluding stopwords such as 'science', 'studies', 'and', etc.).
    - Duplicates across both sources are removed.
    - Final list is sorted alphabetically.
    """
    lower_to_original = {t.lower(): t for t in course_titles}

    result = {}
    for subject in subjects:
        seen_lower = set()
        merged = []
        subject_lower = subject.lower()

        # -- keyword matches --
        for title_lower, original in lower_to_original.items():
            if title_lower == subject_lower:
                continue
            if not shares_keyword(subject_lower, title_lower):
                continue
            if title_lower not in seen_lower:
                merged.append(original)
                seen_lower.add(title_lower)

        # -- manual terms (from subject_mappings.py) --
        for term in manual_mappings.get(subject, []):
            term_lower = term.lower()
            if term_lower == subject_lower:
                continue
            if term_lower not in seen_lower:
                merged.append(term)
                seen_lower.add(term_lower)

        result[subject] = sorted(merged)
    return result


def populate_related_terms_column(mappings):
    """Insert or refresh the related_terms column in course_descriptions.xlsx."""
    df = pd.read_excel(COURSES_FILE)

    related_series = df["subject"].map(
        lambda s: "; ".join(mappings.get(s, []))
    )

    subject_idx = df.columns.get_loc("subject")
    if "related_terms" in df.columns:
        df["related_terms"] = related_series
    else:
        df.insert(subject_idx + 1, "related_terms", related_series)

    df.to_excel(COURSES_FILE, index=False)
    print(f"Updated {COURSES_FILE}")


if __name__ == "__main__":
    print("Gathering course titles from output JSONs...")
    course_titles = gather_course_titles()
    print(f"  {len(course_titles)} unique titles found.\n")

    print("Reading subjects from course_descriptions.xlsx...")
    df = pd.read_excel(COURSES_FILE)
    subjects = df["subject"].dropna().unique().tolist()
    print(f"  {len(subjects)} subjects found.\n")

    print("Loading subject mappings from config/subject_mappings.py...")
    manual = load_subject_mappings()
    manual_count = sum(len(v) for v in manual.values())
    print(f"  {manual_count} manual term(s) loaded.")
    seed_subject_mappings(subjects, manual)
    manual = load_subject_mappings()
    print()

    print("Adding related terms (fuzzy + manual)...\n")
    mappings = add_related_terms(subjects, course_titles, manual)

    for subject in sorted(mappings):
        terms = mappings[subject]
        print(f"{subject}  ({len(terms)} terms)")
        for t in terms:
            print(f"    - {t}")
        print()

    populate_related_terms_column(mappings)
    print("Done.")
