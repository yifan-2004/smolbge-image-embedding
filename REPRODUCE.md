# Reproduce the selected adapter

Use Python 3.12 and a GPU with sufficient memory. Install a Torch/torchvision pair appropriate to your platform, then:

```bash
pip install -r requirements-train.txt
python reproduce.py --device mps
```

`reproduce.py` downloads pinned COCO/Flickr8k/DOCCI sources, verifies ordered image and caption fingerprints against `splits/`, caches frozen features, trains the seed-42 adapter, and evaluates both retrieval directions. It saves outputs under `data/`, `cache/` and `runs/`; allow about 20 GB free disk. Select `--device cuda` on supported NVIDIA hardware.

To reuse existing COCO val2017 files:

```bash
python reproduce.py --device mps \
  --coco-images /path/to/val2017 \
  --coco-captions /path/to/captions_val2017.json
```

Use `--stage cache`, `--stage train` or `--stage evaluate` to resume from a completed stage. `--data`, `--cache` and `--runs` change storage locations. Run additional seeds with `--seeds 42 43 44`. A complete second-machine reproduction has not been claimed; data identities are checked, while backend and library differences may change model numbers slightly.

The image files and caption texts are downloaded from their original sources and are not redistributed here. The published evaluation sets are custom splits; the historical regression sets are already observed. [Data terms](ATTRIBUTION.md) · [Training method](STUDY.md) · [Public metrics](results/evaluation.json).
