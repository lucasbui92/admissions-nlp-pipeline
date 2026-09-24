import argparse, gzip, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from config.schema import DATA_SOURCE
from utils.preprocessing import tokenize_statements_to_sentences

with open(Path("config") / "settings.yml") as f:
    MODE_SETTINGS = yaml.safe_load(f)["modes"]


def main():
    parser = argparse.ArgumentParser(description="Tokenize statements into sentences and cache to pickle.")
    parser.add_argument("--mode", default="sample", choices=["sample", "restricted"])
    args = parser.parse_args()

    mode_cfg = MODE_SETTINGS.get(args.mode, {})
    input_path = mode_cfg.get("input_path")
    input_name = mode_cfg.get("input_name")
    if not input_path or not input_name:
        raise ValueError(
            f"settings.yml is missing input_path/input_name for mode '{args.mode}'. "
            f"Fill these in under modes.{args.mode} before running."
        )

    input_file = Path(input_path) / input_name
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    data_source_type = args.mode

    pkl_path = input_file.parent / f"{input_file.stem}_sentences_tokenized.pkl"

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
