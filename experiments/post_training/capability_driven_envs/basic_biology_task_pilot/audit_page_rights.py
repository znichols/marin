# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify all contributing archived pages and flag possible third-party credits.

This is an evidence inventory, not a legal decision. No absence of a keyword
can certify copyright, permission, or the right to train on an individual page.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from typing import Any

import fsspec
from bs4 import BeautifulSoup

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.finalize import write_new
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.review_full import load_candidates

SOURCE_ARCHIVE_MANIFEST_SHA256 = "9f880299f679ec033b6dd68b422266b7f04a9ecc9ca38ef748b8e6f855b784cd"
REVISION = "gov-bio-archived-page-rights-indicators-v1"
MARKERS = {
    "explicit_copyright": re.compile(r"©|\bcopyright(?:ed)?\b|\ball rights reserved\b", re.I),
    "permission": re.compile(r"used with permission|reproduced with permission|adapted from", re.I),
    "third_party_credit": re.compile(r"courtesy (?:of|:)\s|image credit|photo credit|credited to", re.I),
}


def page_indicators(html: bytes) -> dict[str, int]:
    """Count rights signals in page content without assuming a clean result grants rights."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("main article") or soup.select_one("article")
    if article is None:
        raise ValueError("Archived page is missing an article")
    for node in article.select("script, style, nav, aside, iframe, form"):
        node.decompose()
    text = " ".join(article.get_text(" ", strip=True).split())
    return {name: len(pattern.findall(text)) for name, pattern in MARKERS.items()}


def run(candidates_uri: str, archive_root: str, output_root: str) -> dict[str, Any]:
    """Check all contributing pages against the frozen manifest and write private flags."""
    base = archive_root.rstrip("/") + "/"
    with fsspec.open(base + "manifest.json", "rb") as stream:
        manifest_bytes = stream.read()
    if hashlib.sha256(manifest_bytes).hexdigest() != SOURCE_ARCHIVE_MANIFEST_SHA256:
        raise ValueError("Archive manifest digest mismatch")
    source = json.loads(manifest_bytes)
    candidates = load_candidates(candidates_uri)
    expected = {record["source_url"]: record["source_page_sha256"] for record in candidates}
    if any(expected[record["source_url"]] != record["source_page_sha256"] for record in candidates):
        raise ValueError("Multiple page digests for one source URL")
    if len(expected) != 101:
        raise ValueError(f"Expected 101 contributing articles, found {len(expected)}")
    reviewed: list[dict[str, Any]] = []
    for page in source["pages"]:
        if page["url"] not in expected:
            continue
        if page["sha256"] != expected[page["url"]]:
            raise ValueError(f"Page digest disagrees with candidates: {page['url']}")
        with fsspec.open(base + page["path"], "rb") as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != page["sha256"]:
            raise ValueError(f"Archived page changed: {page['path']}")
        indicators = page_indicators(raw)
        reviewed.append(
            {
                "source_url": page["url"],
                "source_sha256": page["sha256"],
                "source_path": page["path"],
                "indicators": indicators,
                "status": "needs_page_level_rights_adjudication",
                "training_rights_cleared": False,
            }
        )
    if len(reviewed) != len(expected):
        raise ValueError(f"Missing archived candidate pages: {len(reviewed)}/{len(expected)}")
    summary = {
        "revision": REVISION,
        "status": "private_page_indicators_only_not_rights_clearance",
        "source_manifest_sha256": SOURCE_ARCHIVE_MANIFEST_SHA256,
        "contributing_pages": len(reviewed),
        "pages_with_indicators": sum(any(row["indicators"].values()) for row in reviewed),
        "marker_page_counts": dict(
            sorted(Counter(k for row in reviewed for k, count in row["indicators"].items() if count).items())
        ),
        "rights_policy_urls": {name: policy["url"] for name, policy in source["rights_policy_snapshots"].items()},
        "training_rights_cleared": False,
    }
    rows_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviewed).encode()
    summary["page_indicators_sha256"] = hashlib.sha256(rows_bytes).hexdigest()
    output = output_root.rstrip("/") + "/"
    write_new(output + "page-indicators.jsonl", rows_bytes)
    write_new(output + "manifest.json", (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
