# Training protocol

Use the custom image-disjoint COCO and Flickr8k training and development splits recorded in `splits/index.json`. Cache frozen Qwen3-VL image features and frozen BGE text features before training; do not use DOCCI for model selection.

Train a 2048→768→384 MLP with multi-positive contrastive alignment, caption-centroid distillation, a hard-negative margin and image-neighborhood distillation. Balance COCO and Flickr8k at 128 images each per step, use 40 steps per epoch, AdamW at 3e-4, weight decay 0.01, a maximum of 60 epochs and patience 10. Select by the mean development text→image Recall@1 across COCO and Flickr8k. Evaluate fixed weights on the custom regression splits, then on the separate DOCCI confirmation set.
