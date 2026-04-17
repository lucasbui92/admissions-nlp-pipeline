"""
Utility for adding a new column to an Excel file.

If no after_col is specified, the column is appended at the end.
If after_col is specified, the column is inserted directly after it.
The file is updated in place.

Usage (subject example):
    python prep/add_column_excel.py --input <path_to_input.xlsx>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import COURSES_FILE
from config.schema import DATA_SOURCE


def normalize(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def insert_column(file_path, col_name, values, after_col=None):
    """
    Add a new column to an Excel file and save it in place.

    Parameters:
        file_path : path to the Excel file (read and overwritten)
        col_name  : name of the new column
        values    : list of values, one per row
        after_col : column name to insert after; if None, appends at the end
    """
    df = pd.read_excel(file_path)
    df[col_name] = values

    if after_col is not None:
        cols = list(df.columns)
        cols.remove(col_name)
        insert_at = cols.index(after_col) + 1
        cols.insert(insert_at, col_name)
        df = df[cols]

    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df.to_excel(file_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the restricted input Excel file")
    parser.add_argument("--col_name", required=True, help="Name of the new column to add")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    schema = DATA_SOURCE["restricted"]
    course_title_col = schema["course_title"]

    df = pd.read_excel(input_path)
    course_desc_df = pd.read_excel(COURSES_FILE)

    subject_lookup = {
        normalize(s): s
        for s in course_desc_df["subject"].dropna().unique()
    }

    values = [subject_lookup.get(normalize(val)) for val in df[course_title_col]]

    insert_column(input_path, args.col_name, values, after_col=course_title_col)
    print(f"Column '{args.col_name}' written to {input_path}")


if __name__ == "__main__":
    main()
