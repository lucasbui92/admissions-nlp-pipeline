from pathlib import Path


# GLOBAL DATA MODE SWITCH
MODE = "sample"       # change to "restricted" when needed

# Allowed modes
ALLOWED_MODES = {"sample", "restricted"}
if MODE not in ALLOWED_MODES:
    raise ValueError(f"Invalid MODE '{MODE}'. Must be one of {ALLOWED_MODES}")

# BASE DIRECTORIES
DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

DATA_DIR = DATA_ROOT / MODE
OUTPUT_DIR = OUTPUT_ROOT / MODE

# FILE NAME MAPPING PER MODE
INPUT_FILES = {
    "sample": {
        "personal_statements": "sample_personal_statements.xlsx"
    },
    "restricted": {
        "personal_statements": "restricted_personal_statements.xlsx"
    }
}

OUTPUT_FILES = {
    "grammar": "grammar.json",
    "readability": "readability.json"
}

# RESOLVED FULL PATHS (Public Variables)
PERSONAL_STATEMENT_FILE = DATA_DIR / INPUT_FILES[MODE]["personal_statements"]

GRAMMAR_OUTPUT_FILE = OUTPUT_DIR / OUTPUT_FILES["grammar"]
READABILITY_OUTPUT_FILE = OUTPUT_DIR / OUTPUT_FILES["readability"]
