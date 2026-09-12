import json
from pathlib import Path
import chromadb

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "fda_guidance"

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def load_chunks():
    with open(CHUNKS_PATH, "r") as file:
        return json.load(file)

def store_chunks(chunks):
    for chunk in chunks:
        chunk_id = (
            f"{chunk['document_name']}_"
            f"{chunk['page_number']}_"
            f"{chunk['chunk_index']}"
        )

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_hash": chunk["chunk_hash"],
                }
            ],
        )

def embed_all():
    chunks = load_chunks()
    store_chunks(chunks)

    print("Chunks stored:", collection.count())

    return collection.count()
