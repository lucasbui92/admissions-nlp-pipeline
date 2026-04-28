"""
One-time script to populate the Subject column in a restricted input file
by looking up each applicationCourse_titlemain in course_mappings.xlsx.

If the course title is found in course_mappings.xlsx, the corresponding
index is written into the Subject column. Otherwise the cell is left blank.

The input file is modified in-place.

Usage:
    python prep/add_subject_index.py --input <path_to_input_file.xlsx>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import COURSE_MAPPINGS_FILE


def normalize(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def add_subject_index(input_path):
    course_mappings_df = pd.read_excel(COURSE_MAPPINGS_FILE)

    course_lookup = {
        normalize(row["applicationCourse_titlemain"]): row["index"]
        for _, row in course_mappings_df.dropna(
            subset=["applicationCourse_titlemain", "index"]
        ).iterrows()
    }

    df = pd.read_excel(input_path)

    if "applicationCourse_titlemain" not in df.columns:
        print("No applicationCourse_titlemain column found. Nothing to do.")
        return

    if "subject" not in df.columns:
        print("subject column not found. Please add it to the file before running this script.")
        return

    df["subject"] = df["applicationCourse_titlemain"].map(
        lambda v: course_lookup.get(normalize(v))
    )

    matched = df["subject"].notna().sum()
    blank = df["subject"].isna().sum()

    df.to_excel(input_path, index=False)

    print(f"Updated {input_path}")
    print(f"  Matched : {matched}")
    print(f"  Blank   : {blank}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the restricted input Excel file")
    args = parser.parse_args()

    add_subject_index(Path(args.input))
