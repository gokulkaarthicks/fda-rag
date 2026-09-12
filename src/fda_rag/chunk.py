import json
import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = Path("data/processed/clean_pages.json")
OUTPUT_PATH = Path("data/processed/chunks.json")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

def chunk_pages(pages):
    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for index, text in enumerate(page_chunks):
            chunks.append(
                {
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "text": text,
                    "chunk_hash": hash_text(text),
                }
            )

    return chunks

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def chunk_all():
    with open(INPUT_PATH, "r") as file:
        pages = json.load(file)

    chunks = chunk_pages(pages)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            chunks,
            file,
            indent=2,
        )

    print(f"Chunks created: {len(chunks)}")

    return chunks