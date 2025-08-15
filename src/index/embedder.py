from __future__ import annotations

import os
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Force CPU + float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            outputs = self.model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # L2-normalize and return as plain Python lists (no NumPy)
            vecs = F.normalize(pooled, p=2, dim=1)
            return vecs.cpu().tolist()
