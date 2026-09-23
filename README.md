# SmolBGE: direct image embeddings for retrieval

[Model weights](https://huggingface.co/yifanouyang/smolbge-image-embedding) · [Method](STUDY.md) · [Reproduce](REPRODUCE.md)

SmolBGE converts an image directly into a **384-dimensional vector in BGE text space**, without generating a caption. It lets image collections be searched with text queries and used as the image-retrieval component of a multimodal RAG system.

```text
image → frozen Qwen3-VL image encoder → trained MLP → 384-d vector
text  → frozen BGE encoder                       → 384-d vector → cosine retrieval
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

Text→image Recall@1 on custom candidate sets:

| Dataset | Images searched | Recall@1 |
|---|---:|---:|
| COCO | 500 | 67.00% |
| Flickr8k | 1,089 | 68.13% |
| DOCCI | 500 | 57.00% |

Training used **4,000 COCO + 6,000 Flickr8k images and 49,994 descriptions**. DOCCI was used for evaluation only; a separate 500-image DOCCI confirmation set yielded **63.2%** text→image Recall@1. [Evaluation details](results/evaluation.json) · [Split fingerprints](splits/index.json) · [Data attribution](ATTRIBUTION.md).

In one local warmed batch-1 run, direct image encoding averaged **307.9 ms/image** versus **572.0 ms/image** for caption→BGE on the same 96 images over three repeats. These are measurements of one run, not a portable speed guarantee; [timing records](results/latency.json) specify the scope.

[Training code](research_code/train_adapter.py) · [Reproduction](REPRODUCE.md). Code and adapter weights are Apache-2.0; [base-model and data terms](ATTRIBUTION.md) remain separate. OCR, mixed indexes, multilingual queries and RAG answer quality remain untested.
