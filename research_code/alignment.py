"""Loss and development metrics for image-to-BGE alignment."""

import numpy as np
import torch
from torch.nn import functional as F


def read_cache(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def alignment_loss(image, text, mode="multipositive"):
    if mode != "multipositive":
        raise ValueError("Only multi-positive alignment is supported")
    n, k, d = text.shape
    target = F.normalize(text.mean(1), dim=-1)
    raw = image @ target.T
    identity = torch.eye(n, device=image.device, dtype=torch.bool)
    nonpadding = text.reshape(n * k, d).norm(dim=-1) > 0
    flat = text.reshape(n * k, d)[nonpadding]
    owners = torch.arange(n, device=image.device).repeat_interleave(k)[nonpadding]
    logits = image @ flat.T / 0.07
    positives = identity[:, owners]
    image_to_text = (
        torch.logsumexp(logits, 1)
        - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), 1)
    ).mean()
    per_caption = F.cross_entropy(logits.T, owners, reduction="none")
    counts = torch.bincount(owners, minlength=n)
    text_to_image = (per_caption / counts[owners]).sum() / n
    contrastive = 0.5 * (image_to_text + text_to_image)
    negative = raw.masked_fill(identity, -torch.inf).max(1).values
    margin = F.relu(0.1 - raw.diag() + negative).mean()
    distillation = (1 - (image * target).sum(-1)).mean()
    return contrastive + 0.5 * distillation + 0.25 * margin


@torch.inference_mode()
def score_development(model, datasets, device):
    model.eval()
    scores = {}
    for name, data in datasets.items():
        images = torch.cat([
            model(torch.from_numpy(data["vision"][i:i + 512]).to(device)).cpu()
            for i in range(0, len(data["vision"]), 512)
        ]).numpy()
        text = data["text"].reshape(-1, data["text"].shape[-1])
        owners = np.repeat(np.arange(len(images)), data["text"].shape[1])
        nonpadding = np.linalg.norm(text, axis=-1) > 0
        text, owners = text[nonpadding], owners[nonpadding]
        hits = (text @ images.T).argmax(1) == owners
        scores[name] = float(np.mean(np.bincount(owners, weights=hits) / np.bincount(owners)))
    return float(np.mean(list(scores.values()))), scores
