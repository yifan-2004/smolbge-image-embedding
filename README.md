# SmolBGE: direct image embeddings for BGE retrieval

[Download v2 weights](https://huggingface.co/yifanouyang/smolbge-image-embedding/tree/main/v2) · [Method](STUDY.md) · [Reproduce](REPRODUCE.md)

SmolBGE maps images to **384-dimensional BGE text space** without generating captions. v2 uses a frozen Qwen3-VL-Embedding-2B image encoder, a frozen BGE-small text encoder, and a trained 1.87M-parameter adapter. The Hugging Face package contains all three components (~4.1 GiB).

```text
image → frozen image encoder → trained MLP → 384-d vector
text  → frozen BGE encoder               → 384-d vector → cosine retrieval
```

## Use

Python 3.12; sufficient accelerator memory is recommended. The first call downloads the model from Hugging Face.

```bash
pip install -r requirements.txt
python example.py --model yifanouyang/smolbge-image-embedding \
  --images photo.jpg --query "a person riding a bicycle"
```

```python
from model import ImageEmbeddingModel

model = ImageEmbeddingModel.from_pretrained()
image_vectors = model.encode_images(["photo.jpg"])  # [1, 384]
query_vector = model.encode_text("a person riding a bicycle")
scores = image_vectors @ query_vector
```

## Measured results

Text→image Recall@1 (%), with the same custom candidate sets. v2 is the released seed-42 checkpoint. These sets were inspected in earlier rounds, and the change from v0 includes a different image encoder.

| Dataset | Images searched | v0 | v2 |
|---|---:|---:|---:|
| COCO | 500 | 60.76 | **67.00** |
| Flickr8k | 1,089 | 44.67 | **68.13** |
| DOCCI | 500 | 43.40 | **57.00** |

Training used **4,000 COCO + 6,000 Flickr8k images and 49,994 descriptions**. DOCCI was used for evaluation only. A separate 500-image DOCCI confirmation set gave v2 **63.2%** text→image R@1. [Evaluation protocol and metrics](results/evaluation.json) · [Split fingerprints](splits/index.json) · [Data attribution](ATTRIBUTION.md).

On one local device, warmed batch-1 image encoding averaged **307.9 ms** for v2, **572.0 ms** for caption→BGE, and **61.2 ms** for v0 (same 96 images, three repeats). Image decode and model forward are included; model loading, text encoding, vector search and answer generation are excluded. Hardware details are withheld, so these times describe this run and are not a portable speed guarantee. [Timing protocol and records](results/latency.json).

[Training code](research_code/research_round2_train.py) · [Method](STUDY.md) · [Reproduction](REPRODUCE.md). Code and new adapter weights are Apache-2.0; [base-model and data terms](ATTRIBUTION.md) remain separate. OCR, mixed indexes and RAG answer quality remain untested.
