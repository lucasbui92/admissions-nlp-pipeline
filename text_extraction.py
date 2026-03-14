import re
import pandas as pd

from pathlib import Path
from pypdf import PdfReader


def normalize_heading(s):
    return re.sub(r"\s+", " ", s.strip())

BLACKLIST = {
    "HOW TO WRITE A PERSONAL STATEMENT",
    "#IMPATIENTESSEX",
    "CONTENTS",
    "WHAT IS A PERSONAL STATEMENT?",
    "EXAMPLES OF PERSONAL STATEMENTS",
    "HOW TO GET STARTED: PLANNING A PERSONAL STATEMENT",
    "ABOUT THE COURSE",
    "ABOUT THE STUDENT",
    "DO'S AND DON'TS TO TELL YOUR STUDENTS ABOUT WRITING A PERSONAL STATEMENT"
    "EXAMPLES OF PERSONAL STATEMENTS",
    "WHAT NOT TO WRITE: EXAMPLE OF A PERSONAL STATEMENT WITH SEVERAL WEAKNESSES",
    "REFERENCES",
    "INFORMATION FOR TEACHERS AND ADVISORS"
}
BLACKLIST_NORM = {normalize_heading(x) for x in BLACKLIST}

def is_assessment_start(line: str) -> bool:
    s = line.strip()
    s_norm = re.sub(r"\s+", " ", s).lower()
    return s_norm in {"strengths", "weaknesses"}

def extract_full_text(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Normalize common PDF line-break weirdness a bit
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        chunks.append(text)
    return "\n".join(chunks)

def looks_like_subject_heading(line: str) -> bool:
    s = normalize_heading(line)
    if not s:
        return False

    # allow commas & common separators; adjust as needed
    if not re.fullmatch(r"[A-Z0-9&/\-, ]{3,}", s):
        return False

    if len(s) > 80:
        return False

    if s in BLACKLIST_NORM:
        return False
    return True

def split_into_subject_blocks(full_text: str):
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]

    blocks = []
    current_subject = None
    current_statement = []
    skipping_assessment = False
    heading_buf = []

    norm = lambda s: re.sub(r"\s+", " ", s.strip())

    for line in lines:
        # Accumulate consecutive uppercase lines as one heading candidate
        if line.isupper():
            heading_buf.append(line)
            continue

        # Finalize any buffered heading when we hit a non-uppercase line
        if heading_buf:
            candidate_heading = re.sub(r"\s+", " ", " ".join(heading_buf).strip())
            heading_buf = []

            # Detect subject headers
            if looks_like_subject_heading(candidate_heading) and candidate_heading != current_subject:
                if current_subject and current_statement:
                    blocks.append((current_subject, " ".join(current_statement).strip()))
                current_subject, current_statement, skipping_assessment = candidate_heading, [], False

        if current_subject:
            if is_assessment_start(line):
                skipping_assessment = True
            elif not skipping_assessment:
                current_statement.append(line)

    # Handle file ending with an uppercase heading buffered
    if heading_buf:
        candidate_heading = norm(" ".join(heading_buf))
        if looks_like_subject_heading(candidate_heading) and candidate_heading != current_subject:
            if current_subject and current_statement:
                blocks.append((current_subject, " ".join(current_statement).strip()))
            current_subject, current_statement = candidate_heading, []

    if current_subject and current_statement:
        blocks.append((current_subject, " ".join(current_statement).strip()))

    return blocks

def pdf_subject_statements_to_excel(pdf_path: str, out_xlsx: str):
    full_text = extract_full_text(pdf_path)
    subject_blocks = split_into_subject_blocks(full_text)

    rows = []
    for i, (subject, statement) in enumerate(subject_blocks, start=1):
        rows.append(
            {
                "index": i,
                "subject": subject,
                "personal_statement": statement,
            }
        )

    df = pd.DataFrame(rows, columns=["index", "subject", "personal_statement"])
    df.to_excel(out_xlsx, index=False)

    print(f"Done. Extracted {len(df)} subjects -> {out_xlsx}")


if __name__ == "__main__":
    pdf_path = Path("sample_data/Essex sample personal statements.pdf")
    out_xlsx = Path("sample_data/sample_personal_statements.xlsx")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Cannot find PDF: {pdf_path}\n"
            "Put the PDF in sample_data/ or change pdf_path."
        )

    pdf_subject_statements_to_excel(pdf_path, out_xlsx)
