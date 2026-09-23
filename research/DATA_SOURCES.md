# Research data sources and split protocol

The research pipeline adds Flickr8k image-caption training examples and DOCCI
evaluation examples to the existing COCO experiments. These are English natural
image datasets; success here does not establish generality for documents, medical
images, charts, multilingual queries, or arbitrary RAG applications.

| Source | Pinned revision | Intended role | Custom split |
|---|---|---|---|
| [tsystems/flickr8k](https://huggingface.co/datasets/tsystems/flickr8k) | `81fc5f3a41274c80f17b0406426d57cac57ce6fb` | Additional supervised training, development, sealed evaluation | 6,000 train / 1,000 validation / remainder sealed test; seed 2026 |
| [DOCCI official](https://google.github.io/docci/) with [nicolollo/docci Parquet mirror](https://huggingface.co/datasets/nicolollo/docci) | Mirror `d25e433474a72db3853da8e208d564e9b98795e6` | Out-of-domain development and sealed evaluation only | From original 5,000 test images, sample 500 development + disjoint 500 sealed test; seed 2026 |

Flickr8k's pinned mirror contains 8,091 images in three Parquet files (~1.12 GB).
All captions of an image stay together. Byte-identical duplicates, if present,
are collapsed before splitting and their captions are merged. The exact counts
and duplicate records are written into `manifest.json`; without duplicates the
sealed set has 1,091 images. This is **not the official Flickr8k split**.

The first verified preparation found one exact duplicate pair:
`3050606344_af711c726c` and `2851198725_37b6027625`. After merging their captions,
the pre-cross-source-audit split was 6,000 train / 1,000 validation / 1,090 sealed
test. The audit then found that `flickr8k:2947274789_a1a35b33c3` in the new sealed
test set was byte-identical to `coco:110449` in the legacy training set. It is
excluded from the new sealed test manifest, recorded in `exclusions.json`, and
left on disk for provenance. The final split is **6,000 train / 1,000 validation /
1,089 sealed test**, with **29,994 / 5,003 / 5,443** captions respectively.

A second pHash candidate (`coco:160728` / `flickr8k:2583001715_1ce6f58942`, distance
6) was manually inspected and rejected: one image shows a harbor and kayaks; the
other shows a dog jumping for a ball. Both remain in the data. This illustrates
why perceptual-hash candidates are not deleted automatically.

The final audit covers 14,089 retained images (5,000 COCO, 8,089 Flickr8k, and
1,000 DOCCI). It reports zero exact-byte matches across different splits/sources
and only the one rejected cross-source pHash candidate above. The full report is
`artifacts/research/data/leakage_audit.json`. Near-duplicate overlap within one
source and semantic entity overlap are not exhaustively resolved by this audit.

DOCCI contains long human-written descriptions and many subtly related images.
Its original test images are downloaded as six pinned Parquet files (~2.57 GB),
but only the selected 1,000 images are extracted. The original [Google description
JSONL](https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines)
provides the authoritative captions: all 5,000 mirror descriptions must match the
official descriptions by image filename (ignoring only leading/trailing
whitespace). The pipeline fails on a mismatch. No DOCCI row enters a training
manifest. The 500 development rows can be used to diagnose domain transfer, so
they must never be presented as an untouched test set. This custom protocol is
not comparable to scores on the full official 5,000-image test set.

Image IDs and image-byte SHA256 hashes are disjoint between splits. This does not
prove absence of perceptual near-duplicates, shared entities, or overlap with
backbone pretraining. DOCCI authors explicitly describe related images; use its
official cluster/entity metadata for a future stricter cluster-disjoint study.

## Local preparation

```bash
.venv/bin/python scripts/research_prepare_data.py --dataset all --workers 3
```

The script only reads public HTTP data objects and local Parquet files. It never
executes dataset loading scripts and never uses or prints a Hugging Face token.
All downloads and extracted images stay under `artifacts/research/data`.
Pinned Parquet SHA256 hashes are checked against the pinned Hugging Face Git-LFS
object digests. Images retain their original bytes, resolution, and encoding.
The official DOCCI description file is also pinned by SHA256
`c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800`.

Run a cross-source audit against the existing COCO split files:

```bash
.venv/bin/python scripts/research_prepare_data.py --audit-only \
  --audit-coco ../mask-evidence-rs/data/processed/embedding --audit-perceptual
```

This records exact image-byte overlap and cross-source 64-bit DCT perceptual-hash
candidates with Hamming distance at most 6. Review candidates before excluding
anything: perceptual hashes have false positives and false negatives. The audit
does not alter splits, and absence of a match does not establish independence
from backbone pretraining. Pillow, NumPy, and SciPy are required for this optional
audit, and PyArrow is required for Parquet preparation.

Output manifests:

- `artifacts/research/data/flickr8k/train.jsonl`
- `artifacts/research/data/flickr8k/val.jsonl`
- `artifacts/research/data/flickr8k/sealed_test.jsonl`
- `artifacts/research/data/docci/dev.jsonl`
- `artifacts/research/data/docci/sealed_test.jsonl`

Each JSONL row contains `image_id`, absolute `image_file`, `captions`, `source`,
`split`, `original_split`, and `image_sha256`. Corresponding `*_ids.json` files
freeze the image inventories. Paths are machine-specific; rerun preparation to
rebuild local paths while retaining the same image IDs. Use validation for
checkpoint selection, and unlock sealed tests only after committing the model
choice and evaluation configuration. Do not repeatedly tune against sealed scores.

## Attribution and redistribution

The Flickr8k mirror's dataset card labels the mirror MIT; that declaration does
not establish that all underlying Flickr photographs were relicensed as MIT.
Keep the original provenance, publish the preparation code and IDs, and verify
underlying image redistribution rights before republishing image bytes. Cite
Hodosh, Young and Hockenmaier (2013), *Framing Image Description as a Ranking
Task: Data, Models and Evaluation Metrics* ([paper](https://www.jair.org/index.php/jair/article/view/10833)).

Google licenses DOCCI images and annotations under CC BY 4.0 according to the
[official dataset page](https://google.github.io/docci/) and
[official model card](https://huggingface.co/datasets/google/docci). Attribute
Onoe et al. (ECCV 2024), *DOCCI: Descriptions of Connected and Contrasting Images*
([paper](https://arxiv.org/abs/2404.19753)). The mirror is a byte transport source,
not the authority for the data license or captions. Record all source manifests
with research artifacts; do not add raw downloaded images to the code repository.
