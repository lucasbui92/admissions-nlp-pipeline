from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

DEFAULT_INPUT_FILES = {
    "sample": DATA_ROOT / "sample" / "sample_personal_statements.xlsx",
    "restricted": DATA_ROOT / "restricted" / "restricted_personal_statements.xlsx",
}

VALID_MODES = {"sample", "restricted"}


# Runtime IO container
@dataclass
class Paths:
    input_file: Path
    output_dir: Path
    grammar_output_file: Path
    readability_output_file: Path
    data_source_type: str

# Resolver
def resolve_paths(mode: str, external_input: str | None) -> Paths:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

    output_dir = OUTPUT_ROOT / mode

    if external_input:
        input_file = Path(external_input)

        if not input_file.exists():
            raise FileNotFoundError(f"External input file not found: {input_file}")

        data_source_type = "external_raw"
    else:
        input_file = DEFAULT_INPUT_FILES[mode]
        data_source_type = mode

    return Paths(
        input_file=input_file,
        output_dir=output_dir,
        grammar_output_file=output_dir / "grammar.json",
        readability_output_file=output_dir / "readability.json",
        data_source_type=data_source_type,
    )
