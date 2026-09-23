"""Direct image embeddings aligned to frozen BGE-small. No caption generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class VisionProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, output_dim=384, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, value):
        return F.normalize(self.layers(self.input_norm(value)), dim=-1)


class ImageEmbeddingModel:
    """Image/text -> unit BGE vectors from the bundled frozen encoders.

    Query text uses CLS pooling, 128 tokens and no prefix, matching the study.
    The model repository bundles Qwen, the adapter, and BGE. Optional native
    feature methods return the backbone's space, which is not BGE-compatible.
    """

    def __init__(self, directory, device=None, cache_dir=None, local_files_only=False):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "adapter_config.json").read_text())
        self.device = torch.device(device or default_device())
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        if self.config["backbone"] not in {"clip", "siglip2", "qwen"}:
            raise ValueError("Unsupported image backbone")
        filename = self.config["projector_path"]
        if Path(filename).name != filename:
            raise ValueError("Projector must be a file in the model directory")
        weights = self.directory / filename
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.config["projector_sha256"]:
            raise ValueError("Projector checksum mismatch; redownload the release")
        self.projector = VisionProjector(self.config["input_dim"], self.config["hidden_dim"],
                                         self.config["embedding_dim"], self.config["dropout"])
        self.projector.load_state_dict(load_file(str(weights)), strict=True)
        self.projector.to(self.device).eval().requires_grad_(False)
        self.vision_model = self.processor = self.text_model = self.tokenizer = None

    @classmethod
    def from_pretrained(cls, model="yifanouyang/smolbge-image-embedding", *, revision=None,
                        device=None, cache_dir=None, local_files_only=False):
        root = Path(model)
        if not root.is_dir():
            if isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
                raise FileNotFoundError(model)
            root = Path(snapshot_download(str(model), revision=revision, cache_dir=cache_dir,
                        local_files_only=local_files_only,
                        allow_patterns=["adapter_config.json", "projector.safetensors",
                                        "base_models/qwen/**", "base_models/bge/**"]))
        if not (root / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Model adapter_config.json is missing: {root}")
        return cls(root, device, cache_dir, local_files_only)

    def _options(self, config):
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_dir = self.cache_dir
        # Reuse an already pinned default-cache model even when the adapter or
        # another backbone lives in a separate cache. Never change revisions.
        caches = [Path(value) for value in (self.cache_dir, HF_HUB_CACHE) if value]
        for cache in caches:
            snapshot = cache / ("models--" + config["model_id"].replace("/", "--")) / "snapshots" / config["revision"]
            if (snapshot / "config.json").is_file():
                cache_dir = str(cache)
                break
        return dict(revision=config["revision"], cache_dir=cache_dir,
                    local_files_only=self.local_files_only, trust_remote_code=False)

    def _source(self, config):
        bundled = config.get("bundled_path")
        if bundled:
            target = (self.directory / bundled).resolve()
            if not target.is_relative_to(self.directory.resolve()) or not (target / "model.safetensors").is_file():
                raise FileNotFoundError(f"Incomplete bundled model: {bundled}")
            return str(target)
        return config["model_id"]

    def _load_vision(self):
        if self.vision_model is not None:
            return
        config = self.config["vision"]
        source = self._source(config)
        self.processor = AutoProcessor.from_pretrained(source, padding_side="right",
                                                       **self._options(config))
        if self.config["backbone"] == "qwen":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
            dtype = torch.bfloat16 if self.device.type == "mps" or (
                self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            model, loading = Qwen3VLModel.from_pretrained(source, dtype=dtype,
                attn_implementation="sdpa", output_loading_info=True, **self._options(config))
            if loading.get("missing_keys") or loading.get("mismatched_keys") or any(
                    key != "lm_head.weight" for key in loading.get("unexpected_keys", [])):
                raise ValueError("Incomplete Qwen pretrained checkpoint")
        else:
            model = AutoModel.from_pretrained(source, dtype=torch.float32, **self._options(config))
        self.vision_model = model.to(self.device).eval().requires_grad_(False)

    @staticmethod
    def _unit_output(output):
        value = output if isinstance(output, torch.Tensor) else output.pooler_output
        return F.normalize(value.float(), dim=-1)

    def _qwen_batch(self, values, image):
        from qwen_vl_utils import process_vision_info
        config = self.config["vision"]
        conversations = []
        for value in values:
            content = ({"type": "image", "image": "file://" + str(Path(value).resolve()),
                        "min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"]}
                       if image else {"type": "text", "text": value})
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": config["instruction"]}]},
                {"role": "user", "content": [content]},
            ])
        template = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        pictures, videos, kwargs = process_vision_info(conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        if videos is not None:
            raise ValueError("Only images and text are supported by this release")
        try:
            inputs = self.processor(text=template, images=pictures, padding=True, truncation=False,
                                    do_resize=False, return_tensors="pt", **kwargs)
            if int(inputs["attention_mask"].sum(1).max()) > config["max_length"]:
                raise ValueError("Qwen input is too long; split the input instead of truncating visual tokens")
            inputs = inputs.to(self.device)
            hidden = self.vision_model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"]
            last = mask.shape[1] - mask.flip(1).argmax(1) - 1
            return F.normalize(hidden[torch.arange(len(values), device=self.device), last].float(), dim=-1)
        finally:
            for picture in pictures or []:
                picture.close()

    def _image_batch(self, paths):
        if self.config["backbone"] == "qwen":
            return self._qwen_batch(paths, image=True)
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor.image_processor(images=images, return_tensors="pt").to(self.device)
            return self._unit_output(self.vision_model.get_image_features(**inputs))
        finally:
            for image in images:
                image.close()

    @torch.inference_mode()
    def encode_images(self, images, batch_size=1, *, native=False):
        paths = [Path(path) for path in images]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dimension = self.config["input_dim"] if native else self.config["embedding_dim"]
        if not paths:
            return np.empty((0, dimension), dtype=np.float32)
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("All images must be readable local files")
        self._load_vision()
        result = []
        for start in range(0, len(paths), batch_size):
            features = self._image_batch(paths[start:start + batch_size])
            result.append((features if native else self.projector(features)).cpu().numpy())
        return np.concatenate(result)

    @torch.inference_mode()
    def encode_texts(self, texts, batch_size=32, *, native=False):
        values = list(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dimension = self.config["input_dim"] if native else self.config["embedding_dim"]
        if not values:
            return np.empty((0, dimension), dtype=np.float32)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("Texts must be nonempty strings")
        if native:
            self._load_vision()
        elif self.text_model is None:
            config = self.config["text"]
            source = self._source(config)
            self.tokenizer = AutoTokenizer.from_pretrained(source, **self._options(config))
            self.text_model = AutoModel.from_pretrained(source, dtype=torch.float32,
                **self._options(config)).to(self.device).eval().requires_grad_(False)
        result = []
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            if native and self.config["backbone"] == "qwen":
                vectors = self._qwen_batch(batch, image=False)
            elif native:
                maximum = self.vision_model.config.text_config.max_position_embeddings
                tokens = self.processor.tokenizer(batch, padding="max_length", truncation=True,
                                                  max_length=maximum, return_tensors="pt").to(self.device)
                tokens = {key: value for key, value in tokens.items() if key in {"input_ids", "attention_mask"}}
                vectors = self._unit_output(self.vision_model.get_text_features(**tokens))
            else:
                config = self.config["text"]
                tokens = self.tokenizer([config["prefix"] + value for value in batch], padding=True,
                    truncation=True, max_length=config["max_length"], return_tensors="pt").to(self.device)
                vectors = F.normalize(self.text_model(**tokens).last_hidden_state[:, 0].float(), dim=-1)
            result.append(vectors.cpu().numpy())
        return np.concatenate(result)

    def encode_image(self, image):
        return self.encode_images([image])[0]

    def encode_text(self, text):
        return self.encode_texts([text])[0]
