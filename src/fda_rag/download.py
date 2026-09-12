import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

INDEX_URL = (
    "https://www.fda.gov/medical-devices/"
    "digital-health-center-excellence/"
    "guidances-digital-health-content"
)

RAW_DIR = Path("data/raw")

METADATA_PATH = Path("data/documents.json")

def calculate_hash(content):
    return hashlib.sha256(content).hexdigest()

def get_guidance_pages():
    response = requests.get(
        INDEX_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    guidance_pages = {}

    for link in soup.find_all("a", href=True):
        href = link["href"]

        if "/search-fda-guidance-documents/" not in href:
            continue

        title = link.get_text(strip=True)

        if not title:
            continue

        url = urljoin(
            INDEX_URL,
            href,
        )

        guidance_pages[url] = {
            "title": title,
            "url": url,
        }

    return list(
        guidance_pages.values()
    )

def get_pdf_url(guidance_url):
    response = requests.get(
        guidance_url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(" ", strip=True).lower()
        if "download" in text and "guidance" in text:
            return urljoin(guidance_url, href)

    return None

def make_filename(title):
    filename = title.lower()
    filename = re.sub(r"[^a-z0-9]+", "_", filename)
    filename = filename.strip("_")

    return f"{filename}.pdf"

def download_pdf(pdf_url, title):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    filename = make_filename(title)
    output_path = RAW_DIR / filename

    response = requests.get(
        pdf_url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "pdf" not in content_type.lower():
        raise ValueError(
            f"Expected PDF but received {content_type}"
        )


    content = response.content
    document_hash = calculate_hash(content)

    if output_path.exists():
        existing_content = output_path.read_bytes()
        existing_hash = calculate_hash(existing_content)

        if existing_hash == document_hash:
            return output_path, document_hash, False

    output_path.write_bytes(content)

    return output_path, document_hash, True

def load_metadata():
    if not METADATA_PATH.exists():
        return []

    with open(METADATA_PATH, "r") as file:
        return json.load(file)

def save_metadata(documents):
    with open(METADATA_PATH, "w") as file:
        json.dump(documents, file, indent=2)

def build_document_record(
    title,
    guidance_url,
    pdf_url,
    local_path,
    document_hash,
):
    return {
        "title": title,
        "guidance_url": guidance_url,
        "pdf_url": pdf_url,
        "local_path": str(local_path),
        "document_hash": document_hash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

def download_all():
    guidance_pages = get_guidance_pages()

    existing_metadata = load_metadata()

    records_by_url = {
        item["guidance_url"]: item
        for item in existing_metadata
    }

    downloaded = 0
    skipped = 0
    missing = 0

    for document in guidance_pages:
        title = document["title"]
        guidance_url = document["url"]

        pdf_url = get_pdf_url(
            guidance_url
        )

        if not pdf_url:
            missing += 1
            continue

        (
            path,
            document_hash,
            changed,
        ) = download_pdf(
            pdf_url,
            title,
        )

        record = build_document_record(
            title=title,
            guidance_url=guidance_url,
            pdf_url=pdf_url,
            local_path=path,
            document_hash=document_hash,
        )

        records_by_url[
            guidance_url
        ] = record

        if changed:
            downloaded += 1
        else:
            skipped += 1

    records = list(
        records_by_url.values()
    )

    save_metadata(records)

    print(
        f"Download done: {downloaded} new, "
        f"{skipped} unchanged, {missing} missing PDF"
    )

    return records