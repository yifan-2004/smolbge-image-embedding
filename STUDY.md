# Method and evaluation

The model aligns a frozen Qwen3-VL-Embedding-2B image representation with frozen BGE-small-en-v1.5 text vectors through a 2048→768→384 MLP. Only **1,873,024 adapter parameters** were trained. Images are embedded directly without caption generation.

Training uses 4,000 COCO and 6,000 Flickr8k images with 49,994 descriptions. The objective combines multi-positive contrastive learning, caption-centroid distillation, a hard-negative margin and a visual-neighborhood constraint. Seed 42 was chosen after development-set selection. [Split inventories](splits/index.json) and [training code](research_code/train_adapter.py) are provided.

## Retrieval accuracy

Text→image Recall@1 (%) on custom image candidate sets:

| Dataset | Candidates | Recall@1 |
|---|---:|---:|
| COCO | 500 | 67.00 |
| Flickr8k | 1,089 | 68.13 |
| DOCCI | 500 | 57.00 |

These three sets were inspected during development and should be treated as regression measurements. A separate 500-image DOCCI confirmation set was disjoint from the training, development and regression sets. It yielded **63.2% text→image Recall@1** and **57.6% image→text Recall@1**. The model was fixed before that set was measured. The pretrained backbone's exposure to these images is unknown. [Machine-readable metrics](results/evaluation.json).

## Encoding time

One local warmed batch-1 run timed 96 images three times each, interleaving the methods and synchronizing the accelerator. Mean image-encoding time was **307.9 ms/image** for direct embedding and **572.0 ms/image** for caption→BGE. The measurement includes image reading, preprocessing and encoding; it excludes downloads, model loading, text-query encoding, index search and RAG answer generation. Device identity is withheld, so these values are specific to this run. [Per-call records](results/latency.json).

The reported results do not establish OCR, multilingual, mixed-index or end-to-end RAG answer quality. [Source and license notes](ATTRIBUTION.md).
