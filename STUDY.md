# Method and evaluation

The released model takes a frozen Qwen3-VL-Embedding-2B image representation, trains a 2048→768→384 MLP adapter, and aligns its normalized output with frozen BGE-small-en-v1.5 text vectors. Only **1,873,024 adapter parameters** were trained. Images are embedded directly without captions.

Training uses 4,000 COCO and 6,000 Flickr8k images with 49,994 descriptions. The objective combines multi-positive contrastive learning, caption-centroid distillation, a hard-negative margin and a visual-neighborhood constraint. The seed-42 release was fixed after development-set selection. [Exact split inventories](splits/index.json) and [training code](research_code/research_round2_train.py) are provided.

## Retrieval accuracy

Text→image R@1 (%); each row uses the same image candidates for v0 and v2. The three historical sets were observed before the final release, so they are regression measurements.

| Custom split | Candidates | v0 | Released v2 |
|---|---:|---:|---:|
| COCO | 500 | 60.76 | 67.00 |
| Flickr8k | 1,089 | 44.67 | 68.13 |
| DOCCI | 500 | 43.40 | 57.00 |

The later 500-image DOCCI confirmation set was disjoint from training, development and the earlier DOCCI regression set. Released v2 achieved **63.2% text→image R@1** and **57.6% image→text R@1**. The candidate was fixed before this set was measured. The pretrained backbone's exposure to these images is unknown. [Machine-readable public metrics](results/evaluation.json).

## Encoding time

One local warmed batch-1 run timed 96 images three times each, interleaving methods and synchronizing the accelerator. Means: v2 **307.9 ms/image**, caption→BGE **572.0 ms/image**, v0 **61.2 ms/image**. Includes image reading, preprocessing and encoding. Excludes downloads, model loading, text query encoding, index search and RAG answer generation. Device identity is intentionally withheld, so the values are measurements of this run rather than a general throughput claim. [Per-call records](results/latency.json).

These results establish the reported retrieval and timing behavior on the named custom sets. They do not test OCR, multilingual queries, mixed image/text indexes, or final RAG answers. [Source and license notes](ATTRIBUTION.md).
