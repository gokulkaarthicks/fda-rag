from sentence_transformers import CrossEncoder
from fda_rag.retrieve import retrieve

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(MODEL_NAME)

def rerank(query, candidates, top_k=5):
    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )

    return ranked_candidates[:top_k]

def reranked_retrieval(
    query,
    retrieve_k=20,
    final_k=5,
):
    candidates = retrieve(
        query,
        top_k=retrieve_k,
    )

    return rerank(
        query,
        candidates,
        top_k=final_k,
    )