"""Reproduce the selected adapter: prepare -> cache -> train -> evaluate.

Python 3.12; install requirements-train.txt first. Images download from the
original sources and are not redistributed in this model repository.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research_code"))
SPLITS = {"coco": ("train", "validation", "test"),
          "flickr8k": ("train", "val", "sealed_test"), "docci": ("dev", "sealed_test")}
ALIASES = {"val": "validation", "dev": "validation", "sealed_test": "test"}
MODEL_ID = "yifanouyang/smolbge-image-embedding"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(model):
    root = Path(model)
    path = root / "adapter_config.json" if root.is_dir() else Path(
        hf_hub_download(repo_id=str(model), filename="adapter_config.json")
    )
    return json.loads(path.read_text())


def verify_rows(rows, dataset, split):
    expected = read_rows(ROOT / "splits" / f"{dataset}_{split}.jsonl")
    if len(rows) != len(expected):
        raise ValueError("Prepared split has the wrong number of images")
    for row, fingerprint in zip(rows, expected):
        identifier = str(row["image_id"])
        if not identifier.startswith(dataset + ":"):
            identifier = dataset + ":" + identifier
        captions_hash = hashlib.sha256(json.dumps(row["captions"], ensure_ascii=False,
                                                  separators=(",", ":")).encode()).hexdigest()
        if (identifier != fingerprint["image_id"] or captions_hash != fingerprint["captions_sha256"]
                or sha(row["image_file"]) != fingerprint["image_sha256"]):
            raise ValueError(f"Data differs from the published fingerprint: {identifier}")


def prepare(args):
    from research_prepare_data import download, prepare_flickr, prepare_docci
    raw = args.data / "coco/downloads"
    if args.coco_captions:
        annotation = json.loads(args.coco_captions.read_text())
    else:
        annotation_zip = raw / "annotations_trainval2017.zip"
        download("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", annotation_zip)
        with zipfile.ZipFile(annotation_zip) as archive:
            annotation = json.loads(archive.read("annotations/captions_val2017.json"))
    captions = defaultdict(list)
    for item in sorted(annotation["annotations"], key=lambda item: int(item["id"])):
        captions[int(item["image_id"])].append(item["caption"].strip())
    image_root = args.coco_images or args.data / "coco/images"
    image_root.mkdir(parents=True, exist_ok=True)
    expected_ids = [int(row["image_id"].split(":")[-1]) for split in SPLITS["coco"]
                    for row in read_rows(ROOT / "splits" / f"coco_{split}.jsonl")]
    missing = [identifier for identifier in expected_ids if not (image_root / f"{identifier:012d}.jpg").is_file()]
    if missing:
        image_zip = raw / "val2017.zip"
        download("https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", image_zip)
        with zipfile.ZipFile(image_zip) as archive:
            # Extract exact expected basenames only; never extract arbitrary paths.
            for identifier in missing:
                name = f"{identifier:012d}.jpg"
                (image_root / name).write_bytes(archive.read("val2017/" + name))
    for split in SPLITS["coco"]:
        ids = [int(row["image_id"].split(":")[-1]) for row in read_rows(ROOT / "splits" / f"coco_{split}.jsonl")]
        rows = [{"image_id": identifier, "image_file": str((image_root / f"{identifier:012d}.jpg").resolve()),
                 "captions": captions[identifier][:5]} for identifier in ids]
        verify_rows(rows, "coco", split)
        destination = args.data / "coco" / (split + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    prepare_flickr(args.data, 2026, 3)
    prepare_docci(args.data, 2026, 3)
    for dataset, splits in SPLITS.items():
        for split in splits:
            verify_rows(read_rows(args.data / dataset / (split + ".jsonl")), dataset, split)
    print("All 14,089 image and caption fingerprints verified.", flush=True)


def cache(args):
    from model import ImageEmbeddingModel
    model = ImageEmbeddingModel.from_pretrained(args.model, device=args.device)
    backbone = model.config["backbone"]
    batch_size = 2 if backbone == "qwen" else 16
    for dataset, splits in SPLITS.items():
        for split in splits:
            manifest = args.data / dataset / (split + ".jsonl")
            rows = read_rows(manifest)
            verify_rows(rows, dataset, split)
            path = args.cache / backbone / f"{dataset}_{ALIASES.get(split, split)}.npz"
            identity = {"manifest_sha256": sha(manifest), "vision": model.config["vision"],
                        "text": model.config["text"], "encoder_code_sha256": sha(ROOT / "model.py"),
                        "device": str(model.device), "batch_size": batch_size}
            if path.exists():
                old = json.loads(path.with_suffix(".json").read_text())
                if old["identity"] != identity or old["output_sha256"] != sha(path):
                    raise ValueError(f"Stale cache: {path}; choose a new --cache directory")
                continue
            started = time.perf_counter()
            features = []
            for offset in range(0, len(rows), batch_size):
                features.append(model.encode_images([row["image_file"] for row in rows[offset:offset+batch_size]],
                                                     batch_size=batch_size, native=True))
                if offset % 256 == 0:
                    print(dataset, split, offset, "/", len(rows), flush=True)
            captions = [row["captions"] for row in rows]
            flat = [caption for group in captions for caption in group]
            text = model.encode_texts(flat, batch_size=128)
            padded = np.zeros((len(rows), max(map(len, captions)), 384), dtype=np.float32)
            strings = np.full(padded.shape[:2], "", dtype=f"<U{max(map(len, flat))}")
            offset = 0
            for i, group in enumerate(captions):
                padded[i, :len(group)] = text[offset:offset+len(group)]
                strings[i, :len(group)] = group
                offset += len(group)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".partial").open("wb") as stream:
                np.savez(stream, vision=np.concatenate(features), text=padded, captions=strings,
                    image_ids=np.array([str(row["image_id"]) for row in rows]),
                    image_files=np.array([row["image_file"] for row in rows]))
            path.with_suffix(".partial").replace(path)
            path.with_suffix(".json").write_text(json.dumps({"identity": identity, "output_sha256": sha(path),
                                                           "elapsed_s": time.perf_counter()-started}, indent=2))


def train(args):
    config = load_config(args.model)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "research_code")])
    subprocess.run([sys.executable, "-m", "research_code.train_adapter", "--cache", str(args.cache),
        "--output", str(args.runs), "--variants", config["selection"]["variant"],
        "--seeds", *map(str, args.seeds), "--device", args.device], cwd=ROOT, env=environment, check=True)


def evaluate(args):
    import torch
    from model import VisionProjector
    from safetensors.torch import load_file
    from research_metrics import retrieval_metrics
    config = load_config(args.model)
    result = {"role": "custom regression splits", "models": {}}
    for seed in args.seeds:
        variant = config["selection"]["variant"]
        checkpoint = args.runs / f"{variant}_seed{seed}" / "projector.safetensors"
        model = VisionProjector(config["input_dim"]).eval()
        model.load_state_dict(load_file(str(checkpoint)))
        metrics = {}
        for dataset in SPLITS:
            with np.load(args.cache / config["backbone"] / f"{dataset}_test.npz", allow_pickle=False) as data:
                with torch.inference_mode():
                    images = model(torch.from_numpy(data["vision"])).numpy()
                text = data["text"].reshape(-1, 384)
                owner = np.repeat(np.arange(len(images)), data["text"].shape[1])
                keep = np.linalg.norm(text, axis=1) > 0
                entry = retrieval_metrics(images, text[keep], owner[keep])
                entry.pop("per_image")
                metrics[dataset] = entry
        result["models"][f"{variant}_seed{seed}"] = metrics
    args.runs.mkdir(parents=True, exist_ok=True)
    (args.runs / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    from model import default_device
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "cache", "train", "evaluate"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or downloaded model directory")
    parser.add_argument("--coco-images", type=Path)
    parser.add_argument("--coco-captions", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", choices=[42, 43, 44], default=[42])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    args = parser.parse_args()
    for field in ("data", "cache", "runs", "coco_images", "coco_captions"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if args.stage == "all":
        # Separate processes release frozen backbones before adapter training.
        for stage in ("prepare", "cache", "train", "evaluate"):
            command = [sys.executable, str(Path(__file__)), "--stage", stage, "--device", args.device,
                       "--data", str(args.data), "--cache", str(args.cache), "--runs", str(args.runs),
                       "--seeds", *map(str, args.seeds), "--model", args.model]
            if args.coco_images:
                command += ["--coco-images", str(args.coco_images)]
            if args.coco_captions:
                command += ["--coco-captions", str(args.coco_captions)]
            subprocess.run(command, cwd=ROOT, check=True)
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
