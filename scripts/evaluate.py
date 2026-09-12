import json
from pathlib import Path

from fda_rag.rerank import reranked_retrieval
from fda_rag.evaluate import recall_at_k


QUESTIONS_PATH = Path(
    "data/evals/questions.json"
)


def main():
    with open(QUESTIONS_PATH) as file:
        questions = json.load(file)

    scores = []

    for case in questions:
        results = reranked_retrieval(
            case["question"],
            retrieve_k=20,
            final_k=5,
        )

        score = recall_at_k(
            results,
            case["expected_document"],
            case["expected_pages"],
        )

        scores.append(score)

        if score == 0:
            print("\nFAILED")
            print("Question:", case["question"])
            print(
                "Expected:",
                case["expected_document"],
                case["expected_pages"],
            )

            print("Retrieved:")
            for result in results:
                metadata = result["metadata"]
                print(
                    "-",
                    metadata["document_name"],
                    "page",
                    metadata["page_number"],
                )
                
    recall = sum(scores) / len(scores)

    print(f"Recall@5: {recall:.2%}")


if __name__ == "__main__":
    main()