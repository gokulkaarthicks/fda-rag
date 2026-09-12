from pathlib import Path
import hashlib
import pymupdf
import json

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/pages.json")

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for index, page in enumerate(document):
        text = page.get_text("text").strip()

        pages.append(
            {
                "page_number": index + 1,
                "text": text,
                "page_hash": hash_text(text),
            }
        )

    return pages

def parse_all():
    all_pages = []

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        pages = parse_pdf(pdf_path)

        for page in pages:
            page["document_name"] = pdf_path.name

        all_pages.extend(pages)

    PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    with open(PROCESSED_PATH, "w") as file:
        json.dump(all_pages, file, indent=2)

    print("Total pages:", len(all_pages))
    return all_pages