import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import markdown2
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pdfminer.high_level import extract_text

from .cleaning import clean_text


ALLOWED_TYPES = {
    "application/pdf",
    "text/html",
    "text/markdown",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
    "application/json",
}


class UnsupportedFile(Exception):
    pass


def load_file(file_name: str, content: bytes, content_type: str) -> str:
    if content_type not in ALLOWED_TYPES:
        raise UnsupportedFile(f"Unsupported content type {content_type}")

    if content_type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(content)
            temp.flush()
            text = extract_text(temp.name)
        Path(temp.name).unlink(missing_ok=True)
        return clean_text(text)
    if content_type == "text/html":
        soup = BeautifulSoup(content, "html.parser")
        return clean_text(soup.get_text(" "))
    if content_type == "text/markdown":
        return clean_text(markdown2.markdown(content.decode()))
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        temp = io.BytesIO(content)
        doc = DocxDocument(temp)
        return clean_text("\n".join(p.text for p in doc.paragraphs))
    if content_type == "text/csv":
        return clean_text(content.decode())
    if content_type == "application/json":
        parsed = json.loads(content)
        return clean_text(json.dumps(parsed, indent=2))
    return clean_text(content.decode())


def scrape_url(url: str) -> Tuple[str, Dict[str, str]]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else url
    return clean_text(soup.get_text(" ")), {"title": title, "source": url}


def load_git_repo(repo_url: str, branch: str | None = None, path: str | None = None) -> Iterable[Tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        os.system(f"git clone --depth 1 {repo_url} {tmp}")
        repo_path = Path(tmp)
        if branch:
            os.system(f"cd {tmp} && git checkout {branch}")
        base = repo_path / path if path else repo_path
        for file in base.rglob("*"):
            if file.is_file() and file.suffix in {".md", ".txt", ".py", ".html"}:
                yield str(file.relative_to(repo_path)), clean_text(file.read_text(errors="ignore"))


def fetch_api(endpoint: str, headers: Dict[str, str]) -> str:
    response = requests.get(endpoint, headers=headers, timeout=10)
    response.raise_for_status()
    return clean_text(response.text)


def parse_webhook(payload: Dict[str, str]) -> str:
    return clean_text(json.dumps(payload))


def extract_zip(content: bytes) -> Iterable[Tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            yield name, z.read(name)
