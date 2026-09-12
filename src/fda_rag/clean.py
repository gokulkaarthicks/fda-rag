import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/pages.json")
OUTPUT_PATH = Path("data/processed/clean_pages.json")

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line == "Contains Nonbinding Recommendations":
            continue

        if line.isdigit():
            continue

        cleaned_lines.append(line)
    
    text = " ".join(cleaned_lines)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_pages(pages):
    cleaned_pages = []

    for page in pages:
        cleaned_page = page.copy()
        cleaned_page["text"] = clean_text(page["text"])

        cleaned_pages.append(cleaned_page)

    return cleaned_pages

def clean_all():
    with open(INPUT_PATH, "r") as file:
        pages = json.load(file)

    cleaned_pages = clean_pages(pages)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            cleaned_pages,
            file,
            indent=2,
        )

    print(
        f"Pages cleaned: {len(cleaned_pages)}"
    )

    return cleaned_pages