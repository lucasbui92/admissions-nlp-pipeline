import argparse, gzip, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.paths import CACHE_DIR, RAW_DIR, SAMPLE_INPUT_FILE
from config.schema import DATA_SOURCE
from utils.preprocessing import tokenize_statements_to_sentences


def main():
    parser = argparse.ArgumentParser(description="Tokenize statements into sentences and cache to pickle.")
    parser.add_argument("--mode", default="sample", choices=["sample", "restricted"])
    parser.add_argument("--input", default=None, help="Path to input Excel (required for restricted mode)")
    args = parser.parse_args()

    if args.mode == "sample":
        input_file = SAMPLE_INPUT_FILE
        data_source_type = "sample"
    else:
        if not args.input:
            raise ValueError("--input is required for restricted mode")
        input_file = Path(args.input)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        data_source_type = "restricted"

    pkl_path = (RAW_DIR if args.mode == "restricted" else CACHE_DIR) / f"{input_file.stem}_sentences_tokenized.pkl"

    print(f"Pickle path: {pkl_path.resolve()}")

    if pkl_path.exists():
        with gzip.open(pkl_path, "rb") as f:
            sentences = pickle.load(f)
        print(f"Cache hit — {len(sentences)} sentences already saved at {pkl_path}")
        print("Delete the file and re-run to regenerate.")
    else:
        schema = DATA_SOURCE[data_source_type]
        df = pd.read_excel(input_file)

        sentences, stmt_count = tokenize_statements_to_sentences(df, schema)

        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with gzip.open(pkl_path, "wb") as f:
                pickle.dump(sentences, f)
        except Exception:
            pkl_path.unlink(missing_ok=True)
            raise

        print(f"{stmt_count} statements processed → {len(sentences)} sentences saved to {pkl_path}")



if __name__ == "__main__":
    main()
