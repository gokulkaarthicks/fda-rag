from fda_rag.download import download_all
from fda_rag.parse import parse_all
from fda_rag.clean import clean_all
from fda_rag.chunk import chunk_all
from fda_rag.embed import embed_all


def main():
    download_all()
    parse_all()
    clean_all()
    chunk_all()
    embed_all()


if __name__ == "__main__":
    main()