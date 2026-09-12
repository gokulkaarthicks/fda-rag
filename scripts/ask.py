import sys

from fda_rag.generate import ask

def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        return

    query = " ".join(sys.argv[1:])

    try:
        answer = ask(query)
        print("\nANSWER:\n")
        print(answer)

    except Exception as error:
        print(f"Failed: {error}")


if __name__ == "__main__":
    main()