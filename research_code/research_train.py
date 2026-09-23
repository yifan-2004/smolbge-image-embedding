"""Locked multi-seed ablations; only development splits are accessible during training."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from torch.nn import functional as F
from smolbge_image_embedding.modeling import VisionProjector, default_device


def read_cache(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def normalized_centers(text):
    x = text.mean(1)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def alignment_loss(image, text, mode):
    n, k, d = text.shape
    target = F.normalize(text.mean(1), dim=-1)
    raw = image @ target.T
    identity = torch.eye(n, device=image.device, dtype=torch.bool)
    allowed = ~identity
    if mode == 'semantic_mask':
        allowed = allowed & ((target @ target.T) <= 0.90)
    if mode == 'centroid':
        labels = torch.arange(n, device=image.device)
        logits = raw / .07
        contrastive = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    else:
        nonpadding = text.reshape(n*k,d).norm(dim=-1) > 0
        flat = text.reshape(n*k, d)[nonpadding]
        owners = torch.arange(n, device=image.device).repeat_interleave(k)[nonpadding]
        logits = image @ flat.T / .07
        positives = identity[:, owners]
        valid = (allowed | identity)[:, owners]
        logits = logits.masked_fill(~valid, -torch.inf)
        # Probability mass over any paired caption; all k captions remain positives.
        i2t = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), 1)).mean()
        per_caption = F.cross_entropy(logits.T, owners, reduction='none')
        counts = torch.bincount(owners, minlength=n)
        t2i = (per_caption / counts[owners]).sum() / n
        contrastive = .5 * (i2t + t2i)
    negative = raw.masked_fill(~allowed, -torch.inf).max(1).values
    margin = F.relu(.1 - raw.diag() + negative).mean()
    distillation = (1 - (image * target).sum(-1)).mean()
    return contrastive + .5 * distillation + .25 * margin


@torch.inference_mode()
def project(model, features, device):
    model.eval()
    return torch.cat([model(torch.from_numpy(features[i:i+512]).to(device)).cpu()
                      for i in range(0, len(features), 512)]).numpy()


def score_development(model, datasets, device):
    scores = {}
    for name, data in datasets.items():
        images = project(model, data['vision'], device)
        text = data['text'].reshape(-1, data['text'].shape[-1])
        owners = np.repeat(np.arange(len(images)), data['text'].shape[1])
        nonpadding = np.linalg.norm(text, axis=-1) > 0
        text, owners = text[nonpadding], owners[nonpadding]
        # Stable argmax picks the first index on ties. Same actual captions as final evaluation.
        hits = (text @ images.T).argmax(1) == owners
        scores[name] = float(np.mean(np.bincount(owners, weights=hits) / np.bincount(owners)))
    return float(np.mean(list(scores.values()))), scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coco-cache', type=Path, default=Path('../mask-evidence-rs/artifacts/cache/embedding'))
    p.add_argument('--cache', type=Path, default=Path('artifacts/research/cache'))
    p.add_argument('--output', type=Path, default=Path('artifacts/research/runs'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--variants', nargs='+', default=['coco_centroid','mixed_centroid','mixed_multipositive','mixed_semantic_mask'])
    p.add_argument('--epochs', type=int, default=60)
    args = p.parse_args()
    device = default_device()
    print('Training on', device, flush=True)
    torch.set_num_threads(4)
    train_coco = read_cache(args.coco_cache / 'train.npz')
    train_flickr = read_cache(args.cache / 'flickr8k_train.npz')
    development = {'coco': read_cache(args.coco_cache / 'validation.npz'),
                   'flickr8k': read_cache(args.cache / 'flickr8k_validation.npz')}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_hash = hashlib.sha256(Path('research/PROTOCOL.md').read_bytes()).hexdigest()
    data_hashes = {str(path): hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
                   for path in [args.coco_cache/'train.npz',args.coco_cache/'validation.npz',
                                args.cache/'flickr8k_train.npz',args.cache/'flickr8k_validation.npz']}
    for variant in args.variants:
        mode = 'centroid' if variant.endswith('centroid') else ('semantic_mask' if variant.endswith('semantic_mask') else 'multipositive')
        sources = [train_coco] if variant.startswith('coco') else [train_coco, train_flickr]
        max_captions = max(s['text'].shape[1] for s in sources)
        sources = [{**s, 'text': np.pad(s['text'], ((0,0),(0,max_captions-s['text'].shape[1]),(0,0)))} for s in sources]
        for seed in args.seeds:
            out = args.output / f'{variant}_seed{seed}'
            if (out / 'training.json').exists():
                old = json.loads((out / 'training.json').read_text())['config']
                if old.get('protocol_sha256') != protocol_hash or old.get('data_sha256') != data_hashes or old.get('max_epochs') != args.epochs:
                    raise RuntimeError(f'Stale run config/protocol/data: {out}. Use a new run directory.')
                print('Already complete:', out, flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            config = dict(variant=variant, mode=mode, seed=seed, lr=3e-4, weight_decay=.01,
                          batch_size=256, steps_per_epoch=40, max_epochs=args.epochs, patience=10, temperature=.07,
                          distillation_weight=.5, margin_weight=.25, false_negative_threshold=.90,
                          protocol_sha256=protocol_hash, device=str(device), torch=torch.__version__,
                          data_sha256=data_hashes,
                          trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          train_sizes=[len(s['vision']) for s in sources],
                          development_selection='macro actual-caption t2i R@1; COCO+Flickr8k')
            (out / 'config.json').write_text(json.dumps(config, indent=2))
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = VisionProjector().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            best_score, stale, best_state, history = -1., 0, None, []
            start_time = time.perf_counter()
            for epoch in range(1, args.epochs+1):
                model.train()
                # Balanced sources, unique image IDs within a batch. Each source independently shuffles.
                part = 256 // len(sources)
                steps = 40  # Equal update budgets across COCO-only and mixed-data conditions.
                pools = [rng.permutation(len(s['vision'])) for s in sources]
                losses = []
                for step in range(steps):
                    vs, ts = [], []
                    for source, pool in zip(sources, pools):
                        ids = pool[(np.arange(part)+step*part) % len(pool)]
                        vs.append(source['vision'][ids])
                        ts.append(source['text'][ids])
                    v = torch.from_numpy(np.concatenate(vs)).to(device)
                    t = F.normalize(torch.from_numpy(np.concatenate(ts)).to(device), dim=-1)
                    optimizer.zero_grad(set_to_none=True)
                    loss = alignment_loss(model(v), t, mode)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Non-finite loss: {variant} seed={seed} epoch={epoch}')
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                score, individual = score_development(model, development, device)
                item = dict(epoch=epoch, loss=float(np.mean(losses)), development_macro_t2i_r1=score,
                            development_by_dataset=individual, elapsed_s=time.perf_counter()-start_time)
                history.append(item)
                print(variant, seed, json.dumps(item), flush=True)
                if score > best_score+1e-8:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                    save_file({k:v.contiguous() for k,v in best_state.items()}, out / 'projector.safetensors')
                else:
                    stale += 1
                (out / 'progress.json').write_text(json.dumps(history, indent=2))
                if stale >= 10:
                    break
            result = dict(config=config, selected_epoch=best_epoch, development_score=best_score,
                          elapsed_s=time.perf_counter()-start_time, history=history,
                          parameters=sum(x.numel() for x in model.parameters()))
            (out / 'training.json').write_text(json.dumps(result, indent=2))
            print('COMPLETE', out, best_score, flush=True)


if __name__ == '__main__':
    main()
