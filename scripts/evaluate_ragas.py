import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import NodeType
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.utils import num_tokens_from_string

from fda_rag.rerank import reranked_retrieval
from fda_rag.generate import build_context, generate_answer


load_dotenv()

RAW_DIR = Path("data/raw")
MIN_HEADLINE_TOKENS = 500

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")


langchain_llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

ragas_llm = LangchainLLMWrapper(langchain_llm)

langchain_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    langchain_embeddings
)

RAGAS_RUN_CONFIG = RunConfig(max_workers=1)

def _has_headlines_eligible(node) -> bool:
    content = node.properties.get("page_content", "")
    return (
        node.type == NodeType.DOCUMENT
        and num_tokens_from_string(content) > MIN_HEADLINE_TOKENS
    )


def load_documents():
    pages_by_file = defaultdict(list)

    for pdf_path in RAW_DIR.glob("*.pdf"):
        print("Loading:", pdf_path.name)
        pages_by_file[pdf_path.name].extend(PyMuPDFLoader(str(pdf_path)).load())

    documents = []
    for name, pages in pages_by_file.items():
        text = "\n\n".join(page.page_content for page in pages if page.page_content)
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": name},
                )
            )

    print("Documents loaded:", len(documents))
    return documents


def generate_testset(documents):
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    transforms = default_transforms(
        documents=documents,
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    for transform in transforms:
        if isinstance(transform, HeadlineSplitter):
            transform.filter_nodes = _has_headlines_eligible

    print("\nGenerating synthetic questions...")

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=10,
        transforms=transforms,
        run_config=RAGAS_RUN_CONFIG,
    )

    return testset

def answer_questions(testset):
    df = testset.to_pandas()

    responses = []
    retrieved_contexts = []

    for index, row in df.iterrows():

        question = row["user_input"]

        print(f"Question {index + 1}:")
        print(question)

        results = reranked_retrieval(
            question,
            retrieve_k=20,
            final_k=5,
        )

        contexts = [
            result["text"]
            for result in results
        ]

        context = build_context(results)

        answer = generate_answer(
            question,
            context,
        )

        print("\nAnswer:")
        print(answer)

        responses.append(answer)
        retrieved_contexts.append(contexts)

    df["response"] = responses
    df["retrieved_contexts"] = retrieved_contexts

    return df


def main():

    documents = load_documents()

    testset = generate_testset(documents)

    df = answer_questions(testset)

    output_path = Path("data/evals/ragas_testset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved synthetic evaluation dataset:")
    print(output_path)


if __name__ == "__main__":
    main()