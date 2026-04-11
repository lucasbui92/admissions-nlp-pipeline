from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

SAMPLE_INPUT_FILE = DATA_ROOT / "sample" / "sample_personal_statements.xlsx"
COURSES_FILE = DATA_ROOT / "reference" / "course_descriptions.xlsx"

VALID_MODES = {"sample", "restricted"}


@dataclass
class Paths:
    input_file: Path
    reference_file: Path
    output_dir: Path
    grammar_output_file: Path
    readability_output_file: Path
    semantic_output_file: Path
    chunk_semantic_output_file: Path
    data_source_type: str


def resolve_paths(mode, external_input, output_name):
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

    if mode == "sample":
        input_file = SAMPLE_INPUT_FILE
        data_source_type = "sample"
    else:
        if not external_input:
            raise ValueError("Restricted mode requires --input")
        input_file = Path(external_input)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        data_source_type = "restricted"

    if not COURSES_FILE.exists():
        raise FileNotFoundError(f"Reference file not found: {COURSES_FILE}")

    today = datetime.now().strftime("%Y%m%d")
    output_dir = OUTPUT_ROOT / mode / f"{output_name}_{today}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        input_file=input_file,
        reference_file=COURSES_FILE,
        output_dir=output_dir,
        grammar_output_file=output_dir / "grammar.json",
        readability_output_file=output_dir / "readability.json",
        semantic_output_file=output_dir / "semantic.json",
        chunk_semantic_output_file=output_dir / "chunk_semantic.json",
        data_source_type=data_source_type,
    )
