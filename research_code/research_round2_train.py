"""Round-two frozen-encoder alignment and native-neighborhood distillation.

Only COCO/Flickr train and validation caches are read. Incomplete runs restart
from their fixed seed; completed runs are reused only when all fingerprints
match. Selection requires the entire six-variant, three-seed matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

import research_train as round1
from research_train import alignment_loss, score_development
from smolbge_image_embedding import modeling
from smolbge_image_embedding.modeling import VisionProjector, default_device


VARIANTS = (
    "clip_align", "clip_geometry", "siglip2_align", "siglip2_geometry",
    "qwen_align", "qwen_geometry",
)
SEEDS = (42, 43, 44)
CACHE_COUNTS = {"coco_train": 4000, "coco_validation": 500,
                "flickr8k_train": 6000, "flickr8k_validation": 1000}
MAX_EPOCHS = 60
STEPS_PER_EPOCH = 40
BATCH_SIZE = 256
PATIENCE = 10
GEOMETRY_TEMPERATURE = 0.1
GEOMETRY_WEIGHT = 0.25


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def geometry_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Mean row KL(teacher-neighbors || student-neighbors), excluding self.

    Both feature sets may have different dimensions. Teacher features detach;
    diagonal entries are removed rather than masked with infinity, so neither
    zero-times-infinity nor undefined singleton softmaxes can occur. This is a
    query-independent relationship objective, not caption or query distillation.
    """
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError("student and teacher must be matrices with matching batch size")
    if not student.shape[0] or not student.shape[1] or not teacher.shape[1]:
        raise ValueError("feature matrices must be nonempty")
    if student.device != teacher.device:
        raise ValueError("student and teacher must be on the same device")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student.shape[0] == 1:
        return student.sum() * 0.0
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    n = student.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=student.device)
    teacher_logits = (teacher @ teacher.T)[off_diagonal].reshape(n, n - 1) / temperature
    student_logits = (student @ student.T)[off_diagonal].reshape(n, n - 1) / temperature
    teacher_logprob = F.log_softmax(teacher_logits, dim=-1)
    student_logprob = F.log_softmax(student_logits, dim=-1)
    return (teacher_logprob.exp() * (teacher_logprob - student_logprob)).sum(dim=-1).mean()


def training_loss(projected, text, native_features, use_geometry: bool):
    """Return (total, unchanged_round1_alignment, unweighted_geometry)."""
    alignment = alignment_loss(projected, text, "multipositive")
    geometry = geometry_loss(projected, native_features, GEOMETRY_TEMPERATURE) if use_geometry else projected.sum() * 0.0
    return alignment + GEOMETRY_WEIGHT * geometry, alignment, geometry


def validate_cache(data: dict, label: str, expected_images: int) -> dict:
    for key in ("vision", "text", "image_ids"):
        if key not in data:
            raise ValueError(f"{label}: missing {key}")
    vision = np.asarray(data["vision"], dtype=np.float32)
    text = np.asarray(data["text"], dtype=np.float32)
    ids = np.asarray(data["image_ids"]).astype(str)
    if vision.ndim != 2 or vision.shape[0] != expected_images or vision.shape[1] < 1:
        raise ValueError(f"{label}: wrong vision shape {vision.shape}")
    if text.ndim != 3 or text.shape[0] != expected_images or text.shape[1] < 1 or text.shape[2] != 384:
        raise ValueError(f"{label}: text must have shape [images, captions, 384]")
    if ids.shape != (expected_images,) or len(set(ids.tolist())) != expected_images:
        raise ValueError(f"{label}: image IDs must be unique and match cache rows")
    if not np.isfinite(vision).all() or not np.isfinite(text).all():
        raise ValueError(f"{label}: non-finite cached features")
    if (np.linalg.norm(vision, axis=-1) == 0).any():
        raise ValueError(f"{label}: zero native image feature")
    norms = np.linalg.norm(text, axis=-1)
    present = norms > 0
    if not present.any(axis=1).all():
        raise ValueError(f"{label}: every image needs a nonpadding caption")
    if not np.allclose(norms[present], 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{label}: expected normalized BGE caption features")
    return {**data, "vision": vision, "text": text, "image_ids": ids}


def load_encoder_data(root: Path, encoder: str) -> dict:
    loaded, data_hashes, metadata_hashes = {}, {}, {}
    for name, count in CACHE_COUNTS.items():
        path = root / encoder / f"{name}.npz"
        loaded[name] = validate_cache(round1.read_cache(path), f"{encoder}/{name}", count)
        data_hashes[f"{encoder}/{name}.npz"] = sha256(path)
        if path.with_suffix(".json").exists():
            metadata_hashes[f"{encoder}/{name}.json"] = sha256(path.with_suffix(".json"))
    dimensions = {data["vision"].shape[1] for data in loaded.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{encoder}: native feature dimensions disagree between splits")
    for source in ("coco", "flickr8k"):
        train_ids = set(loaded[f"{source}_train"]["image_ids"].tolist())
        validation_ids = set(loaded[f"{source}_validation"]["image_ids"].tolist())
        if train_ids & validation_ids:
            raise ValueError(f"{encoder}/{source}: train/validation image-ID overlap")
    return {"loaded": loaded, "input_dim": dimensions.pop(),
            "data_sha256": data_hashes, "cache_metadata_sha256": metadata_hashes}


def check_same_labels(reference: dict, candidate: dict) -> None:
    """Changing the encoder must not silently change image order or BGE targets."""
    for key in CACHE_COUNTS:
        left, right = reference["loaded"][key], candidate["loaded"][key]
        if not np.array_equal(left["image_ids"], right["image_ids"]):
            raise ValueError(f"Encoder caches disagree on image IDs/order: {key}")
        if left["text"].shape != right["text"].shape or not np.allclose(left["text"], right["text"], atol=1e-6, rtol=1e-6):
            raise ValueError(f"Encoder caches disagree on frozen BGE targets: {key}")


def validate_resume_config(recorded: dict, expected: dict) -> None:
    """Strict guard includes imported Round1 loss and projector implementation."""
    if recorded != expected:
        changed = sorted(key for key in recorded.keys() | expected.keys() if recorded.get(key) != expected.get(key))
        raise RuntimeError(f"Stale run configuration ({', '.join(changed)}); use a new output directory")


def run_config(variant: str, seed: int, data: dict, fingerprints: dict, device: torch.device) -> dict:
    encoder = variant.rsplit("_", 1)[0]
    dim = data["input_dim"]
    # LayerNorm(D), Linear(D,768), Linear(768,384), with biases.
    parameters = 2 * dim + dim * 768 + 768 + 768 * 384 + 384
    return {
        "variant": variant, "encoder": encoder, "seed": seed,
        "encoder_family": "full_vlm_native_image_embedding" if encoder == "qwen" else "native_dual_encoder_image_embedding",
        "mode": "multipositive", "use_geometry": variant.endswith("_geometry"),
        "input_dim": dim, "hidden_dim": 768, "output_dim": 384,
        "parameters": parameters, "dropout": 0.1,
        "lr": 3e-4, "weight_decay": 0.01, "batch_size": BATCH_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "gradient_clip": 1.0,
        "alignment_temperature": 0.07, "distillation_weight": 0.5,
        "margin_weight": 0.25, "margin": 0.1,
        "geometry_temperature": GEOMETRY_TEMPERATURE,
        "geometry_weight": GEOMETRY_WEIGHT if variant.endswith("_geometry") else 0.0,
        "train_sizes": [4000, 6000], "development_sizes": {"coco": 500, "flickr8k": 1000},
        "development_selection": "macro actual-caption t2i R@1; COCO+Flickr8k",
        "teacher": "detached native image-only representation; no query input",
        "encoder_comparison_scope": "system/representation comparison; Qwen also changes full VLM processing",
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "data_sha256": data["data_sha256"], "cache_metadata_sha256": data["cache_metadata_sha256"],
        **fingerprints,
    }


def train_run(output: Path, config: dict, data: dict, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path, completed = output / "config.json", output / "training.json"
    if config_path.exists():
        validate_resume_config(json.loads(config_path.read_text()), config)
    if completed.exists():
        record = json.loads(completed.read_text())
        validate_resume_config(record["config"], config)
        if sha256(output / "projector.safetensors") != record["weights_sha256"]:
            raise RuntimeError(f"Completed run weights changed: {output}")
        print("Already complete:", output, flush=True)
        return
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    seed = config["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VisionProjector(input_dim=config["input_dim"], hidden_dim=768, output_dim=384, dropout=0.1).to(device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != config["parameters"]:
        raise RuntimeError("Projector parameter count does not match saved configuration")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sources = [data["loaded"]["coco_train"], data["loaded"]["flickr8k_train"]]
    max_captions = max(source["text"].shape[1] for source in sources)
    sources = [{**source, "text": np.pad(source["text"], ((0, 0), (0, max_captions-source["text"].shape[1]), (0, 0)))}
               for source in sources]
    development = {source: data["loaded"][f"{source}_validation"] for source in ("coco", "flickr8k")}
    history, best_score, best_epoch, stale = [], -1.0, None, 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        pools = [rng.permutation(len(source["vision"])) for source in sources]
        totals, alignments, geometries = [], [], []
        for step in range(STEPS_PER_EPOCH):
            visual_parts, text_parts = [], []
            for source, pool in zip(sources, pools):
                indices = pool[(np.arange(BATCH_SIZE // 2) + step * (BATCH_SIZE // 2)) % len(pool)]
                visual_parts.append(source["vision"][indices])
                text_parts.append(source["text"][indices])
            vision = torch.from_numpy(np.concatenate(visual_parts)).to(device)
            text = F.normalize(torch.from_numpy(np.concatenate(text_parts)).to(device), dim=-1)
            optimizer.zero_grad(set_to_none=True)
            total, alignment, geometry = training_loss(model(vision), text, vision, config["use_geometry"])
            if not bool(torch.isfinite(torch.stack((total, alignment, geometry))).all()):
                raise FloatingPointError(f"Non-finite loss: {config['variant']} seed={seed} epoch={epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals.append(float(total.detach()))
            alignments.append(float(alignment.detach()))
            geometries.append(float(geometry.detach()))
        score, individual = score_development(model, development, device)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite development score")
        item = {"epoch": epoch, "loss": float(np.mean(totals)),
                "alignment_loss": float(np.mean(alignments)), "geometry_loss": float(np.mean(geometries)),
                "development_macro_t2i_r1": score, "development_by_dataset": individual,
                "elapsed_s": time.perf_counter() - started}
        history.append(item)
        print(config["variant"], seed, json.dumps(item), flush=True)
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                      output / "projector.safetensors")
        else:
            stale += 1
        (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= PATIENCE:
            break
    result = {"config": config, "selected_epoch": best_epoch, "development_score": best_score,
              "elapsed_s": time.perf_counter() - started, "history": history,
              "parameters": actual_parameters, "optimizer_updates": len(history) * STEPS_PER_EPOCH,
              "weights_sha256": sha256(output / "projector.safetensors")}
    (output / "training.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("COMPLETE", output, best_score, flush=True)


def write_selection_if_complete(output: Path, expected_configs: dict) -> dict | None:
    keys = [f"{variant}_seed{seed}" for variant in VARIANTS for seed in SEEDS]
    if any(not (output / key / "training.json").exists() for key in keys):
        return None
    scores, weights = {}, {}
    for variant in VARIANTS:
        values = []
        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            record = json.loads((output / key / "training.json").read_text())
            validate_resume_config(record["config"], expected_configs[key])
            score = float(record["development_score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite development score: {key}")
            values.append(score)
            weight_hash = sha256(output / key / "projector.safetensors")
            if weight_hash != record["weights_sha256"]:
                raise RuntimeError(f"Saved weights do not match completed run: {key}")
            weights[key] = weight_hash
        scores[variant] = float(np.mean(values))
    selected = max(VARIANTS, key=lambda variant: scores[variant])
    selection = {"criterion": "highest three-seed mean COCO/Flickr development macro t2i R@1",
                 "variants": list(VARIANTS), "seeds": list(SEEDS), "completed_runs": len(keys),
                 "development_scores": scores, "selected_variant": selected, "representative_seed": 42,
                 "protocol_sha256": expected_configs[keys[0]]["protocol_sha256"],
                 "weights_sha256": weights,
                 "note": "Selected without loading tests; ties follow preregistered variant order."}
    (output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/research/round2/cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research/round2/runs"))
    parser.add_argument("--protocol", type=Path, default=Path("research/round2/PROTOCOL.md"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = default_device() if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(4)
    fingerprints = {"protocol_sha256": sha256(args.protocol), "trainer_sha256": sha256(Path(__file__)),
                    "dependencies_sha256": {"research_train.py": sha256(Path(round1.__file__)),
                                            "modeling.py": sha256(Path(modeling.__file__))}}
    loaded, expected_configs = {}, {}

    def data_for(encoder):
        if encoder not in loaded:
            data = load_encoder_data(args.cache, encoder)
            if loaded:
                check_same_labels(next(iter(loaded.values())), data)
            loaded[encoder] = data
        return loaded[encoder]

    # Validate every requested cache before starting accelerator training.
    for variant in args.variants:
        data_for(variant.rsplit("_", 1)[0])
    print("TRAINING DEVICE", device, flush=True)
    for variant in args.variants:
        data = data_for(variant.rsplit("_", 1)[0])
        for seed in args.seeds:
            key = f"{variant}_seed{seed}"
            config = run_config(variant, seed, data, fingerprints, device)
            expected_configs[key] = config
            train_run(args.output / key, config, data, device)
    all_keys = [f"{variant}_seed{seed}" for variant in VARIANTS for seed in SEEDS]
    if all((args.output / key / "training.json").exists() for key in all_keys):
        for variant in VARIANTS:
            data = data_for(variant.rsplit("_", 1)[0])
            for seed in SEEDS:
                expected_configs[f"{variant}_seed{seed}"] = run_config(variant, seed, data, fingerprints, device)
        selection = write_selection_if_complete(args.output, expected_configs)
        print("DEVELOPMENT SELECTION", json.dumps(selection), flush=True)
    else:
        print("Matrix incomplete; no final selection was written.", flush=True)


if __name__ == "__main__":
    main()
