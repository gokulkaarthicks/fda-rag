import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from fda_rag.rerank import reranked_retrieval

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def build_context(results):
    context_parts = []

    for result in results:
        metadata = result["metadata"]

        context_parts.append((
            f"SOURCE: {metadata['document_name']}\n"
            f"PAGE: {metadata['page_number']}\n\n"
            f"{result['text']}"
        ))

    return "\n\n---\n\n".join(context_parts)

def generate_answer(query, context):
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the supplied FDA guidance context. "
                    "If the context is insufficient, say so. "
                    "Cite the source document and page."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT: {context}\n"
                    f"QUESTION: {query}"
                ),
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def ask(query):
    results = reranked_retrieval(
        query,
        retrieve_k=20,
        final_k=5,
    )

    context = build_context(results)

    return generate_answer(
        query,
        context,
    )