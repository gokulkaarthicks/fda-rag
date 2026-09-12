def recall_at_k(
    results,
    expected_document,
    expected_pages,
):
    for result in results:
        metadata = result["metadata"]

        if (
            metadata["document_name"] == expected_document
            and metadata["page_number"] in expected_pages
        ):
            return 1

    return 0