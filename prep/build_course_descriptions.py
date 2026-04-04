import pandas as pd

from config.paths import COURSES_FILE

SOURCES = [
    ("Essex", "description_essex"),
    ("Manchester", "description_manchester"),
    ("UCAS", "description_ucas"),
]


def build_combined_description(course_desc_df):
    def merge_row(row):
        parts = []
        for label, col in SOURCES:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                parts.append(f"{label}: {str(value).strip()}")
        return "\n".join(parts) if parts else None

    course_desc_df = course_desc_df.copy()
    course_desc_df["combined_description"] = course_desc_df.apply(merge_row, axis=1)
    return course_desc_df


if __name__ == "__main__":
    df = pd.read_excel(COURSES_FILE)
    df = build_combined_description(df)
    df.to_excel(COURSES_FILE, index=False)
    print(f"Written combined_description to {COURSES_FILE}")
