#!/usr/bin/env python3
"""Prepare pinned Flickr8k training data and held-out DOCCI research data.

Downloads are public HTTP objects; no remote dataset Python or credentials run.
Image bytes are copied without resizing or re-encoding. Evaluation captions never
enter the training manifest. Run with this project's Python containing pyarrow.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


FLICKR_REVISION = "81fc5f3a41274c80f17b0406426d57cac57ce6fb"
DOCCI_REVISION = "d25e433474a72db3853da8e208d564e9b98795e6"
DOCCI_ANNOTATIONS = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DOCCI_ANNOTATION_SHA256 = "c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800"
# Frozen before any new-model test evaluation; the old released adapter saw this COCO training image.
KNOWN_COCO_TRAIN_OVERLAP = {
    "b4f1012183b47482a31955ac38bbad9b7711b018ff20e2cb60b355033f323dc6": "coco:110449"
}
FLICKR_SHARDS = [
    ("train-00000-of-00003.parquet", 373214970, "5bb630f4389fb1d01705577dcc7769a6fcdcc5598be47b11f5995f6351a7b866"),
    ("train-00001-of-00003.parquet", 363960832, "8a47954a325853d95c481261fbc0dec2fea06ae21267545ea7ae03ec390a057e"),
    ("train-00002-of-00003.parquet", 378814640, "22e0d386d2896db64d68b6e1855133a353809ad22a204543edb3583f0280c9e5"),
]
DOCCI_SHARDS = [
    ("test-00000-of-00006.parquet", 434153061, "2cc70816eda9c39d93f51d50e56873cd24f4fb550f42166c0d7ccf19cb0bd888"),
    ("test-00001-of-00006.parquet", 416627779, "5b3107748198f73d092d84cbaea56f7c66e05afbf739b6222ca8aed080566c67"),
    ("test-00002-of-00006.parquet", 425828960, "229795b5f0dd6f999b81a8895eb0fec37c4c15e6b6b03b22f51bba14a6ac3fd5"),
    ("test-00003-of-00006.parquet", 435018351, "1f2925fbd1aded5ef708953538f83ff6a39e46ed33229a4368033bfa541772a4"),
    ("test-00004-of-00006.parquet", 425266864, "9606bceb67960d97b39f72bcfac2c0c2b2d811ca8df2ce6a1d4ab7180d1805ff"),
    ("test-00005-of-00006.parquet", 436398070, "02cb9bc93a9c352e63cbf9c810e4399f5e7b9e0da58ce68a0ea654a5477ded52"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, size: int | None = None, digest: str | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (size is None or destination.stat().st_size == size):
        actual = sha256_file(destination)
        if digest is None or actual == digest:
            print(f"Verified existing {destination.name}", flush=True)
            return {"url": url, "file": str(destination), "bytes": destination.stat().st_size, "sha256": actual}
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 5):
        try:
            request = Request(url, headers={"User-Agent": "SmolBGE-research-data/1.0"})
            with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                downloaded = 0
                last_message = time.monotonic()
                while block := response.read(4 * 1024 * 1024):
                    handle.write(block)
                    downloaded += len(block)
                    if time.monotonic() - last_message > 20:
                        print(f"Downloading {destination.name}: {downloaded / 1e6:.1f} MB", flush=True)
                        last_message = time.monotonic()
            if size is not None and downloaded != size:
                raise ValueError(f"Size mismatch for {destination.name}: {downloaded} != {size}")
            actual = sha256_file(partial)
            if digest is not None and actual != digest:
                raise ValueError(f"SHA256 mismatch for {destination.name}")
            partial.replace(destination)
            print(f"Downloaded and verified {destination.name}: {downloaded / 1e6:.1f} MB", flush=True)
            return {"url": url, "file": str(destination), "bytes": downloaded, "sha256": actual}
        except Exception as error:
            if attempt == 4:
                raise
            print(f"Retry {attempt}/3 {destination.name}: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("Unreachable")


def fetch_shards(root: Path, repo: str, revision: str, shards: list, workers: int) -> list[dict]:
    def fetch(shard: tuple) -> dict:
        filename, size, digest = shard
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{filename}?download=true"
        return download(url, root / "downloads" / filename, size, digest)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fetch, shards))


def iter_rows(shards: list[dict]):
    for shard in shards:
        print(f"Reading {Path(shard['file']).name}", flush=True)
        for batch in pq.ParquetFile(shard["file"]).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def write_image(root: Path, image_id: str, image: dict) -> tuple[str, str]:
    content = image.get("bytes")
    if not isinstance(content, bytes) or not content:
        raise ValueError(f"Missing embedded image bytes: {image_id}")
    digest = hashlib.sha256(content).hexdigest()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.get("path") or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    image_path = image_dir / f"{image_id.split(':')[-1]}{suffix}"
    if not image_path.exists() or sha256_file(image_path) != digest:
        image_path.write_bytes(content)
    return str(image_path.resolve()), digest


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_split(root: Path, split: str, rows: list[dict]) -> None:
    path = root / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "split": split}, ensure_ascii=False) + "\n")
    write_json(root / f"{split}_ids.json", [row["image_id"] for row in rows])
    print(f"Prepared {path}: {len(rows)} images, {sum(len(row['captions']) for row in rows)} captions", flush=True)


def prepare_flickr(root: Path, seed: int, workers: int) -> dict:
    root = root / "flickr8k"
    shards = fetch_shards(root, "tsystems/flickr8k", FLICKR_REVISION, FLICKR_SHARDS, workers)
    rows = []
    seen_ids = set()
    seen_hashes = {}
    duplicates = []
    for item in iter_rows(shards):
        image_id = f"flickr8k:{Path(item['image_filename']).stem}"
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image ID {image_id}")
        seen_ids.add(image_id)
        image_file, digest = write_image(root, image_id, item["image"])
        captions = list(dict.fromkeys(text.strip() for text in item["captions"] if text.strip()))
        if not captions:
            raise ValueError(f"Missing captions {image_id}")
        if digest in seen_hashes:
            original = seen_hashes[digest]
            original["captions"] = list(dict.fromkeys(original["captions"] + captions))
            duplicates.append({"excluded": image_id, "retained": original["image_id"]})
            continue
        row = {"image_id": image_id, "image_file": image_file, "captions": captions,
               "source": "flickr8k", "image_sha256": digest, "original_split": "train"}
        seen_hashes[digest] = row
        rows.append(row)
    if len(seen_ids) != 8091:
        raise ValueError(f"Unexpected source image count: {len(seen_ids)}")
    rows.sort(key=lambda row: row["image_id"])
    random.Random(seed).shuffle(rows)
    if len(rows) <= 7000:
        raise ValueError("Too few unique images after duplicate exclusion")
    splits = {"train": rows[:6000], "val": rows[6000:7000], "sealed_test": rows[7000:]}
    exclusions = []
    for name in ["val", "sealed_test"]:
        retained = []
        for row in splits[name]:
            if row["image_sha256"] in KNOWN_COCO_TRAIN_OVERLAP:
                exclusions.append({"image_id": row["image_id"], "split": name,
                                   "image_sha256": row["image_sha256"],
                                   "overlap_with": KNOWN_COCO_TRAIN_OVERLAP[row["image_sha256"]],
                                   "reason": "Exact image bytes already used in legacy COCO training"})
            else:
                retained.append(row)
        splits[name] = retained
    for name, selected in splits.items():
        save_split(root, name, selected)
    write_json(root / "exclusions.json", exclusions)
    manifest = {"dataset": "flickr8k", "repository": "tsystems/flickr8k", "revision": FLICKR_REVISION,
                "seed": seed, "split_type": "custom_image_disjoint_not_official",
                "source_images": len(seen_ids), "unique_images": len(rows), "exact_duplicates": duplicates,
                "cross_source_exclusions": exclusions,
                "counts": {name: len(selected) for name, selected in splits.items()}, "downloads": shards,
                "sha256": {name: sha256_file(root / f"{name}.jsonl") for name in splits}}
    write_json(root / "manifest.json", manifest)
    return manifest


def prepare_docci(root: Path, seed: int, workers: int) -> dict:
    root = root / "docci"
    annotations = download(DOCCI_ANNOTATIONS, root / "downloads" / "docci_descriptions.jsonlines",
                           11000214, DOCCI_ANNOTATION_SHA256)
    official = {}
    with Path(annotations["file"]).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "test":
                official[Path(record["image_file"]).stem] = record
    if len(official) != 5000:
        raise ValueError(f"Expected 5000 official test images; got {len(official)}")
    shuffled = sorted(official)
    random.Random(seed).shuffle(shuffled)
    selected_split = {image_id: "dev" for image_id in shuffled[:500]}
    selected_split.update({image_id: "sealed_test" for image_id in shuffled[500:1000]})
    shards = fetch_shards(root, "nicolollo/docci", DOCCI_REVISION, DOCCI_SHARDS, workers)
    selected = {"dev": [], "sealed_test": []}
    seen_ids = set()
    seen_hashes = {}
    for item in iter_rows(shards):
        image = item["image_file"]
        image_id = Path(image.get("path") or "").stem
        if image_id not in official:
            raise ValueError(f"Mirror image ID not found in official annotations: {image_id}")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate mirror ID: {image_id}")
        seen_ids.add(image_id)
        if item["description"].strip() != official[image_id]["description"].strip():
            raise ValueError(f"Mirror description differs from Google official annotation: {image_id}")
        if image_id not in selected_split:
            continue
        namespaced_id = f"docci:{image_id}"
        image_file, digest = write_image(root, namespaced_id, image)
        if digest in seen_hashes:
            raise ValueError(f"Duplicate image bytes between selected DOCCI samples: {image_id}, {seen_hashes[digest]}")
        seen_hashes[digest] = image_id
        split = selected_split[image_id]
        selected[split].append({"image_id": namespaced_id, "image_file": image_file,
                                "captions": [official[image_id]["description"]], "source": "docci",
                                "image_sha256": digest, "original_split": "test"})
    if seen_ids != set(official):
        raise ValueError("Mirror ID inventory differs from official test set")
    for name, rows in selected.items():
        rows.sort(key=lambda row: row["image_id"])
        if len(rows) != 500:
            raise ValueError(f"Expected 500 {name} rows; got {len(rows)}")
        save_split(root, name, rows)
    manifest = {"dataset": "docci", "repository": "nicolollo/docci", "revision": DOCCI_REVISION,
                "official_source": DOCCI_ANNOTATIONS, "seed": seed,
                "split_type": "custom_500_dev_500_sealed_from_original_5000_test_not_official",
                "usage": "evaluation_only_never_training", "official_descriptions_verified": len(seen_ids),
                "counts": {name: len(rows) for name, rows in selected.items()},
                "annotations": annotations, "downloads": shards,
                "sha256": {name: sha256_file(root / f"{name}.jsonl") for name in selected}}
    write_json(root / "manifest.json", manifest)
    return manifest


def audit_cross_dataset(root: Path, coco_root: Path, perceptual: bool = False) -> dict:
    """Report overlap without changing any split or automatically deleting data."""
    from itertools import combinations

    manifests = []
    for source, directory, names in [
        ("coco", coco_root, ["train", "validation", "test"]),
        ("flickr8k", root / "flickr8k", ["train", "val", "sealed_test"]),
        ("docci", root / "docci", ["dev", "sealed_test"]),
    ]:
        for split in names:
            path = directory / f"{split}.jsonl"
            if path.exists():
                manifests.append((source, split, path))
    fingerprints = []
    for source, split, manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                image_id = str(item["image_id"])
                if not image_id.startswith(source + ":"):
                    image_id = source + ":" + image_id
                path = Path(item["image_file"])
                row = {"image_id": image_id, "source": source, "split": split,
                       "image_sha256": item.get("image_sha256") or sha256_file(path)}
                if perceptual:
                    import numpy as np
                    from PIL import Image, ImageOps
                    from scipy.fft import dctn

                    with Image.open(path) as image:
                        gray = ImageOps.exif_transpose(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
                        coefficients = dctn(np.asarray(gray, dtype=np.float32), type=2, norm="ortho")[:8, :8].ravel()
                        bits = coefficients > np.median(coefficients[1:])
                        value = 0
                        for bit in bits:
                            value = (value << 1) | bool(bit)
                        row["phash64"] = f"{value:016x}"
                fingerprints.append(row)
        print(f"Audited {source}/{split}", flush=True)
    by_digest = {}
    exact_matches = []
    for row in fingerprints:
        for previous in by_digest.get(row["image_sha256"], []):
            if (previous["source"], previous["split"]) != (row["source"], row["split"]):
                exact_matches.append({"left": previous, "right": row})
        by_digest.setdefault(row["image_sha256"], []).append(row)
    perceptual_candidates = []
    if perceptual:
        sources = sorted({row["source"] for row in fingerprints})
        for left_source, right_source in combinations(sources, 2):
            left_rows = [(row, int(row["phash64"], 16)) for row in fingerprints if row["source"] == left_source]
            right_rows = [(row, int(row["phash64"], 16)) for row in fingerprints if row["source"] == right_source]
            for left, left_hash in left_rows:
                for right, right_hash in right_rows:
                    distance = (left_hash ^ right_hash).bit_count()
                    if distance <= 6:
                        perceptual_candidates.append({"left": left, "right": right, "phash_hamming": distance})
            print(f"Compared perceptual hashes: {left_source} vs {right_source}", flush=True)
    counts = {}
    for row in fingerprints:
        key = row["source"] + "/" + row["split"]
        counts[key] = counts.get(key, 0) + 1
    summary = {"counts": counts, "exact_matches": exact_matches,
               "perceptual_method": "64-bit DCT pHash, cross-source Hamming <= 6" if perceptual else None,
               "perceptual_candidates": perceptual_candidates,
               "interpretation": "Candidates require manual review; no images/splits modified. Byte hash cannot identify resized copies. Perceptual matches can be false positives; no match does not prove no semantic overlap."}
    summary["manual_perceptual_reviews"] = [{
        "left": "coco:160728", "right": "flickr8k:2583001715_1ce6f58942", "phash_hamming": 6,
        "decision": "false_positive_keep_both",
        "evidence": "Manual full-image inspection: COCO is a harbor with people and kayaks; Flickr is a dog jumping for a ball on grass.",
    }]
    write_json(root / "leakage_audit.json", summary)
    with (root / "image_fingerprints.jsonl").open("w", encoding="utf-8") as handle:
        for row in fingerprints:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"audit_counts": counts, "exact_matches": len(exact_matches),
                      "perceptual_candidates": len(perceptual_candidates)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickr8k", "docci", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path("artifacts/research/data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--audit-coco", type=Path, help="Directory containing existing COCO train/validation/test JSONL")
    parser.add_argument("--audit-only", action="store_true", help="Skip downloads and only run the requested overlap audit")
    parser.add_argument("--audit-perceptual", action="store_true", help="Also report cross-source DCT pHash candidates")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.audit_only and not args.audit_coco:
        parser.error("--audit-only requires --audit-coco")
    if not args.audit_only and args.dataset in {"flickr8k", "all"}:
        result = prepare_flickr(root, args.seed, args.workers)
        print(json.dumps({"dataset": "flickr8k", "counts": result["counts"]}), flush=True)
    if not args.audit_only and args.dataset in {"docci", "all"}:
        result = prepare_docci(root, args.seed, args.workers)
        print(json.dumps({"dataset": "docci", "counts": result["counts"]}), flush=True)
    if args.audit_coco:
        audit_cross_dataset(root, args.audit_coco.resolve(), args.audit_perceptual)


if __name__ == "__main__":
    main()
