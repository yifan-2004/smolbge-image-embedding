# Sources and attribution

## Frozen models

- [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32), OpenAI.
- [SigLIP 2 base patch16 224](https://huggingface.co/google/siglip2-base-patch16-224), Google.
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), Qwen.
  Qwen processing follows the official chat format and last-valid-token pooling,
  using the upstream `qwen-vl-utils` library. This study caps image pixels at
  262,144 and sequence length at 8,192; it is not the official full-resolution
  leaderboard configuration. Qwen is already retrieval-trained.
- [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), BAAI.
- [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
  Hugging Face, used by the older adapter and caption-latency baseline.

The release bundles the exact pinned Qwen and BGE checkpoints under
`base_models/`, alongside our trained adapter. Qwen remains Apache-2.0 and
BGE remains MIT; attribution and original terms remain with those files.
Project code is Apache-2.0; our adapter has separate terms in
`WEIGHTS_LICENSE.md`. These licenses do not relicense training datasets or
certify commercial clearance for the trained model.

## Datasets

COCO val2017 captions, custom 4,000/500/500 split; Flickr8k, custom
6,000/1,000/1,089 split after recorded duplicate handling; DOCCI, custom
500-development/500-regression subsets from the original test collection,
plus a later 500-image confirmation from previously unused test IDs.

- [COCO](https://cocodataset.org/#download), Lin et al.; caption annotations and
  original source photographs retain their respective terms.
- Flickr8k: Hodosh, Young and Hockenmaier, *Framing Image Description as a Ranking
  Task* ([paper](https://www.jair.org/index.php/jair/article/view/10833)); bytes from
  the pinned [tsystems mirror](https://huggingface.co/datasets/tsystems/flickr8k).
  A mirror license label does not establish relicensing rights for all Flickr photographs.
  The [original dataset page](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html)
  provides access for non-commercial research and educational purposes. This
  study does not establish commercial clearance for a model trained on those
  images; do not infer such clearance from the mirror's license label.
- [DOCCI](https://google.github.io/docci/), Onoe et al., ECCV 2024; original
  images/annotations CC BY 4.0. The pinned
  [nicolollo mirror](https://huggingface.co/datasets/nicolollo/docci) supplies
  transport files; official annotation descriptions are checked separately.

Only ordered IDs and content fingerprints are included here. See
[`splits/index.json`](splits/index.json) for exact source URLs, revisions and hashes.

## Closely related methods

This is an empirical reproduction/adapter study, not a claim that frozen towers,
MLPs, contrastive learning or neighborhood distillation are new.

- [Freeze-Align, CVPR 2025](https://github.com/mayug/freeze-align): frozen unimodal
  encoders aligned through lightweight projectors.
- [VISTA / Visualized-BGE, ACL 2024](https://arxiv.org/abs/2406.04292): extending
  text embeddings to visual and mixed-modal retrieval.
- [Probabilistic Knowledge Transfer, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf)
  and [Relational Knowledge Distillation, CVPR 2019](https://arxiv.org/abs/1904.05068):
  earlier relationship-preserving distillation frameworks.
- [MKP-Adapter](https://arxiv.org/abs/2609.16875): a particularly close recent
  precedent for adapter-based backward compatibility and knowledge preservation.
  Our image-to-fixed-text-space setting is not evidence of novelty by itself.

For differences, confounds and falsifiable follow-up questions, see the
[interpretation guide](research/round2/INTERPRETATION_GUIDE.md).
