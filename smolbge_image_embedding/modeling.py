from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer


class VisionProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(self.input_norm(features)), dim=-1)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_attention_mask(pixel_attention_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    masks = pixel_attention_mask.view(-1, *pixel_attention_mask.shape[-2:])
    patches = masks.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return patches.sum(dim=(-1, -2)).gt(0)


def masked_patch_pool(hidden: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    flat_mask = patch_mask.flatten(1).to(hidden.dtype).unsqueeze(-1)
    return (hidden * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1)


class SmolBGEEmbedder:
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.config = json.loads((self.model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        self.device = torch.device(device) if device is not None else default_device()
        use_bf16 = self.device.type == "mps" or (
            self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.vision_dtype = torch.bfloat16 if use_bf16 else torch.float32

        self.projector = VisionProjector(
            input_dim=int(self.config["vision_dim"]),
            hidden_dim=int(self.config["hidden_dim"]),
            output_dim=int(self.config["embedding_dim"]),
            dropout=float(self.config["dropout"]),
        )
        self.projector.load_state_dict(load_file(str(self.model_dir / self.config["projector_path"])))
        self.projector = self.projector.to(self.device).eval()

        self.image_processor = AutoProcessor.from_pretrained(self.config["vision_model"]).image_processor
        self.image_processor.do_image_splitting = False
        vision_container = AutoModelForMultimodalLM.from_pretrained(
            self.config["vision_model"], dtype=self.vision_dtype
        )
        self.vision_model = vision_container.model.vision_model.to(self.device).eval()
        self.vision_config = vision_container.config.vision_config
        del vision_container

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["text_model"])
        self.text_model = AutoModel.from_pretrained(
            self.config["text_model"], dtype=torch.float32
        ).to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path = "yifanouyang/smolbge-image-embedding",
        device: str | torch.device | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "SmolBGEEmbedder":
        root = Path(model_dir)
        if not root.exists():
            if isinstance(model_dir, Path) or str(model_dir).startswith((".", "/", "~")):
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
            root = Path(snapshot_download(
                repo_id=str(model_dir), revision=revision,
                local_files_only=local_files_only,
                allow_patterns=["model/adapter_config.json", "weights/projector.safetensors"],
            ))
        if not (root / "adapter_config.json").exists() and (root / "model" / "adapter_config.json").exists():
            root = root / "model"
        if not (root / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No adapter_config.json found in {root}")
        config = json.loads((root / "adapter_config.json").read_text())
        weights = root / config["projector_path"]
        if not weights.is_file():
            raise FileNotFoundError("Adapter weights missing. Run git lfs pull or use from_pretrained() to download from Hugging Face.")
        with weights.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Weights are a Git LFS pointer. Run git lfs pull or use from_pretrained() to download from Hugging Face.")
        return cls(root, device=device)

    @torch.inference_mode()
    def encode_images(
        self,
        images: Iterable[str | Path | Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:
        items = list(images)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not items:
            return np.empty((0, int(self.config["embedding_dim"])), dtype=np.float32)
        outputs = []
        for start in range(0, len(items), batch_size):
            opened = [
                item.convert("RGB") if isinstance(item, Image.Image) else Image.open(item).convert("RGB")
                for item in items[start : start + batch_size]
            ]
            try:
                inputs = self.image_processor(images=opened, do_image_splitting=False, return_tensors="pt")
                pixels = inputs["pixel_values"].view(-1, *inputs["pixel_values"].shape[-3:]).to(
                    device=self.device, dtype=self.vision_dtype
                )
                pixel_mask = inputs["pixel_attention_mask"].to(self.device)
                patch_mask = patch_attention_mask(pixel_mask, int(self.vision_config.patch_size))
                hidden = self.vision_model(
                    pixel_values=pixels,
                    patch_attention_mask=patch_mask,
                    return_dict=True,
                ).last_hidden_state
                outputs.append(self.projector(masked_patch_pool(hidden.float(), patch_mask)).cpu())
            finally:
                for image in opened:
                    image.close()
        return torch.cat(outputs).numpy()

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Iterable[str],
        is_query: bool = False,
        batch_size: int = 128,
    ) -> np.ndarray:
        items = list(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if is_query:
            prefix = self.config.get("query_prefix", "")
            items = [prefix + text for text in items]
        if not items:
            return np.empty((0, int(self.config["embedding_dim"])), dtype=np.float32)
        outputs = []
        for start in range(0, len(items), batch_size):
            tokens = self.tokenizer(
                items[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            cls = self.text_model(**tokens).last_hidden_state[:, 0]
            outputs.append(F.normalize(cls.float(), dim=-1).cpu())
        return torch.cat(outputs).numpy()

    def encode_image(self, image: str | Path | Image.Image) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode_texts([text], is_query=is_query)[0]
