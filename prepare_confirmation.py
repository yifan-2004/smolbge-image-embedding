"""Create the fixed unused 500-image DOCCI confirmation set after reproduce.py --stage prepare."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

code = Path(__file__).resolve().parent / "research_code"
if not code.is_dir():
    code = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(code))
from research_prepare_data import iter_rows, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/docci"))
    args = parser.parse_args()
    root = args.data
    official = {}
    for line in (root / "downloads/docci_descriptions.jsonlines").read_text().splitlines():
        record = json.loads(line)
        if record["split"] == "test":
            official[Path(record["image_file"]).stem] = record
    if len(official) != 5000:
        raise ValueError("Official DOCCI test inventory is incomplete")
    ids = sorted(official)
    random.Random(2026).shuffle(ids)
    wanted = set(ids[1000:1500])
    earlier = set()
    earlier_bytes = set()
    for name in ("dev", "sealed_test"):
        for line in (root / f"{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            earlier.add(row["image_id"].removeprefix("docci:"))
            earlier_bytes.add(row["image_sha256"])
    if wanted & earlier:
        raise ValueError("Confirmation IDs overlap earlier evaluations")
    source = json.loads((root / "manifest.json").read_text())
    rows = []
    byte_hashes = set()
    for item in iter_rows(source["downloads"]):
        picture = item["image_file"]
        image_id = Path(picture.get("path") or "").stem
        if image_id not in wanted:
            continue
        if item["description"].strip() != official[image_id]["description"].strip():
            raise ValueError(f"Official caption differs: {image_id}")
        filename, digest = write_image(root, "docci:" + image_id, picture)
        if digest in earlier_bytes or digest in byte_hashes:
            raise ValueError(f"Duplicate image bytes: {image_id}")
        byte_hashes.add(digest)
        rows.append({"image_id": "docci:" + image_id, "image_file": filename,
                     "image_sha256": digest, "captions": [official[image_id]["description"]]})
    rows.sort(key=lambda row: row["image_id"])
    if len(rows) != 500:
        raise ValueError("Expected 500 confirmation images")
    output = root / "confirmation_500.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    fingerprint = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(rows), "manifest": str(output), "sha256": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
