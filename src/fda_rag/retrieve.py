import json
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
import re

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "fda_guidance"

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

with open(CHUNKS_PATH, "r") as file:
    chunks = json.load(file)


def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )

tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


def dense_search(query, top_k=10):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    items = []

    for i in range(len(results["ids"][0])):
        items.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "rank": i + 1,
            }
        )

    # print("Items:", items)
    return items

def bm25_search(query, top_k=10):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        chunk_id = (
            f"{chunk['document_name']}_"
            f"{chunk['page_number']}_"
            f"{chunk['chunk_index']}"
        )

        results.append(
            {
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": {
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                },
                "rank": rank,
            }
        )
    # print("Results:", results)
    return results

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
):
    scores = {}
    items = {}

    for result in dense_results:
        chunk_id = result["id"]

        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += 1 / (k + result["rank"])

        items[chunk_id] = result

    for result in bm25_results:
        chunk_id = result["id"]

        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += 1 / (k + result["rank"])

        items[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        {
            **items[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids
    ]

def retrieve(query, top_k=20):
    dense_results = dense_search(
        query,
        top_k=top_k,
    )

    bm25_results = bm25_search(
        query,
        top_k=top_k,
    )

    return reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )[:top_k]