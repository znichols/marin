# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect public HF access/license signals for ID leads without reading any questions.

This is evidence for human rights review, not automated permission to acquire
items or generate training tasks. One source can supply several variants.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import CATALOG_COMMIT, sha256

HF_API = "https://huggingface.co/api/datasets/"
REPO_ROOT = Path(__file__).resolve().parents[4]


def hf_repo(url: str) -> str | None:
    """Extract only the public dataset repository identifier from a catalog URL."""
    parts = urlparse(url)
    path = parts.path.strip("/").split("/")
    if parts.netloc != "huggingface.co" or len(path) < 3 or path[0] != "datasets":
        return None
    return "/".join(path[1:3])


def listed_hf_repos(inventory: dict[str, Any]) -> dict[str, list[str]]:
    """Identify unique dataset repos while preserving which historical variants link them."""
    repos: dict[str, set[str]] = {}
    for row in inventory["rows"]:
        for source in row["catalog_sources"]:
            repo = hf_repo(source["url"])
            if repo:
                repos.setdefault(repo, set()).add(row["family"] + "/" + row["variant"])
    return {repo: sorted(variants) for repo, variants in sorted(repos.items())}


def probe(inventory: dict[str, Any]) -> dict[str, Any]:
    """Read only HF repository metadata; refuse to label source/item rights cleared."""
    if inventory["catalog_commit"] != CATALOG_COMMIT or inventory["historical_id_rows"] != 46:
        raise ValueError("Input must be the pinned ID catalog inventory")
    records = []
    for repo, variants in listed_hf_repos(inventory).items():
        response = requests.get(HF_API + repo, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        records.append(
            {
                "repo": repo,
                "variants": variants,
                "revision_observed": metadata["sha"],
                "declared_license": metadata.get("cardData", {}).get("license"),
                "gated": metadata.get("gated"),
                "private": metadata.get("private"),
                "card_url": f"https://huggingface.co/datasets/{repo}/blob/{metadata['sha']}/README.md",
                "item_rights_reviewed": False,
                "upstream_rights_reviewed": False,
                "training_from_eval_items_permitted": False,
            }
        )
    return {
        "status": "metadata_only_rights_signals_not_approval",
        "catalog_commit": CATALOG_COMMIT,
        "hf_repositories": len(records),
        "license_declarations": dict(sorted(Counter(str(row["declared_license"]) for row in records).items())),
        "access_states": dict(sorted(Counter(str(row["gated"]) for row in records).items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private path outside Git")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Rights evidence must stay outside the Git checkout")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = probe(json.loads(args.inventory.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, sort_keys=True))
    print(f"probe_sha256={sha256(args.output.read_bytes())}")


if __name__ == "__main__":
    main()
