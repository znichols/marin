"""Extract bounded, text-only government biology passages from the frozen raw archive."""

import argparse
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

EXPECTED_MANIFEST_SHA256 = "9f880299f679ec033b6dd68b422266b7f04a9ecc9ca38ef748b8e6f855b784cd"
MAX_CHARS = 3200
STOP_HEADINGS = (
    "for more information",
    "topics in the",
    "related contents",
    "related content",
    "scientific journal article",
    "references",
    "further reading",
)


def extract_body(html: bytes, *, source: str) -> list[str]:
    """Extract article text, leaving images, linked documents, and site chrome out."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("main article" if source == "nhgri" else "article")
    if article is None:
        raise ValueError(f"No article body in {source} page")
    for element in article.select("script, style, nav, aside, figure, figcaption, img, video, iframe, form"):
        element.decompose()
    paragraphs: list[str] = []
    for element in article.find_all(("h1", "h2", "h3", "p", "li", "tr")):
        if not isinstance(element, Tag) or (element.find_parent("li") and element.name == "p"):
            continue
        if element.name == "li" and element.find(("h1", "h2", "h3", "p", "tr")):
            continue
        if element.name == "tr" and element.find_parent("li"):
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if element.name in ("h1", "h2", "h3") and text.lower().startswith(STOP_HEADINGS):
            break
        if not text or (element.name == "li" and len(text) > 1000):
            continue
        paragraphs.append(text)
    body = " ".join(paragraphs).lower()
    if "©" in body or "all rights reserved" in body or "used with permission" in body:
        raise ValueError("Page-level copyright marker in extracted body: quarantine for rights review")
    return paragraphs


def chunks(paragraphs: list[str]) -> list[dict[str, object]]:
    """Split at paragraph boundaries, retaining indices for source tracing."""
    output: list[dict[str, object]] = []
    current: list[str] = []
    first = 0
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph) > MAX_CHARS:
            raise ValueError(f"A single paragraph exceeds {MAX_CHARS} characters")
        if current and sum(len(p) + 1 for p in current) + len(paragraph) > MAX_CHARS:
            output.append({"first_paragraph": first, "last_paragraph": index - 1, "text": "\n".join(current)})
            current = []
            first = index
        current.append(paragraph)
    if current:
        output.append({"first_paragraph": first, "last_paragraph": len(paragraphs) - 1, "text": "\n".join(current)})
    return output


def prepare(root: Path) -> list[dict[str, object]]:
    """Verify raw bytes and enumerate content-addressed passages (no model calls)."""
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Wrong raw-source manifest")
    manifest = json.loads(raw)
    rows: list[dict[str, object]] = []
    for page in manifest["pages"]:
        path = str(page["path"])
        parts = Path(path).parts
        source = parts[1]
        if (source == "medlineplus" and len(parts) != 5) or (source == "nhgri" and path.endswith("/index.html")):
            continue
        html = (root / path).read_bytes()
        if hashlib.sha256(html).hexdigest() != page["sha256"]:
            raise ValueError(f"Source bytes changed: {path}")
        for chunk in chunks(extract_body(html, source=source)):
            text = str(chunk["text"])
            if len(text) < 200:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            rows.append(
                {
                    "id": f"{source}-{digest[:20]}",
                    "source_family": source,
                    "source_url": page["url"],
                    "source_page_sha256": page["sha256"],
                    "source_path": path,
                    "passage_sha256": digest,
                    **chunk,
                }
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate passage identity")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"Prepared {len(rows)} passages across {len({r['source_url'] for r in rows})} articles")


if __name__ == "__main__":
    main()
