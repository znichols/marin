# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze metadata-only historical-ID leads from the biology eval catalog.

An imported spreadsheet ID label is not a reviewed training permission or even
necessarily a confirmed ID assignment for the catalog's current variant.
This script does not fetch or output benchmark items.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

CATALOG_COMMIT = "c4f5c922a11acc68c0b70b46da8367a2a37a92cd"
SOURCE_INVENTORY_SHA256 = "1c4c7c768c08e507cf82aef017c89e7bd52bf55463d9257454f543c45844b2a3"
EXPECTED_HISTORICAL_ID_ROWS = 46
REPO_ROOT = Path(__file__).resolve().parents[4]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def historical_id_rows(source_inventory: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Join each old ID variant to the corresponding current variant without promoting its ID status."""
    rows = []
    for original in source_inventory["records"]:
        if original["assignment"] != "ID":
            continue
        family = catalog[original["family_id"]]
        variants = [v for v in family["variants"] if v["id"] == original["variant_id"]]
        if len(variants) != 1:
            raise ValueError(f"Missing/ambiguous catalog variant: {original['family_id']}/{original['variant_id']}")
        variant = variants[0]
        rows.append(
            {
                "family": original["family_id"],
                "variant": original["variant_id"],
                "name": original["name"],
                "historical_assignment": "ID",
                "catalog_assignment": variant["assignment"],
                "selected_variant": (family.get("selected_variant") or family["variants"][0]["id"]) == variant["id"],
                "catalog_sources": variant["sources"],
                "historical_sources": original["sources"],
                "training_rights_status": "not_reviewed",
                "eval_access_status": "not_reviewed",
                "upstream_rights_status": "not_reviewed",
            }
        )
    return sorted(rows, key=lambda row: (row["family"], row["variant"]))


def inventory(catalog_root: Path) -> dict[str, Any]:
    """Fail if the local checkout or frozen spreadsheet projection differs from the audited pin."""
    revision = subprocess.check_output(["git", "-C", str(catalog_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != CATALOG_COMMIT:
        raise ValueError(f"Catalog revision {revision} differs from pin {CATALOG_COMMIT}")
    dirty = subprocess.check_output(["git", "-C", str(catalog_root), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("Catalog checkout has uncommitted changes")
    raw = (catalog_root / "data/source-inventory.json").read_bytes()
    if sha256(raw) != SOURCE_INVENTORY_SHA256:
        raise ValueError("Historical source inventory bytes changed")
    catalog = {p.stem: yaml.safe_load(p.read_text()) for p in (catalog_root / "benchmarks").glob("*.yaml")}
    rows = historical_id_rows(json.loads(raw), catalog)
    if len(rows) != EXPECTED_HISTORICAL_ID_ROWS or len({(r["family"], r["variant"]) for r in rows}) != len(rows):
        raise ValueError("Unexpected historical ID coverage")
    return {
        "catalog_commit": revision,
        "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
        "historical_id_rows": len(rows),
        "families": len({row["family"] for row in rows}),
        "current_assignment_counts": dict(sorted(Counter(row["catalog_assignment"] for row in rows).items())),
        "status": "metadata_only_not_training_clearance",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root", type=Path, required=True, help="Clean checkout of pinned Open-Athena/bio-evals"
    )
    parser.add_argument("--output", type=Path, required=True, help="Private path outside the source tree")
    args = parser.parse_args()
    result = inventory(args.catalog_root)
    if args.output.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Private inventory must stay outside the Git checkout")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))
    print(f"inventory_sha256={sha256(args.output.read_bytes())}")


if __name__ == "__main__":
    main()
