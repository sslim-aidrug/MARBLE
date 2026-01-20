"""ChemBERTa drug encoder component."""

from typing import Any, Dict, Iterable, Optional

import torch
from transformers import AutoModel, RobertaTokenizer

from components.base import BaseDrugEncoder


class DrugEncoder(BaseDrugEncoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        pretrained_name = config.get("pretrained_name", "seyonec/ChemBERTa-zinc-base-v1")
        freeze_embeddings = config.get("freeze_embeddings", True)
        freeze_layers = config.get("freeze_layers", 6)

        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        self.encoder = AutoModel.from_pretrained(pretrained_name)

        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        if freeze_layers:
            for layer in self.encoder.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.output_dim = int(self.encoder.config.hidden_size)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def forward(self, inputs: Any) -> torch.Tensor:
        return self.encoder(**inputs).last_hidden_state[:, 0]


class DrugDataLoader:
    """Tokenizer wrapper for registry-based data loading."""

    def __init__(self, config: Dict[str, Any], base_path: Optional[str] = None):
        _ = base_path
        self.config = config
        model_cfg = config.get("model", {}).get("drug_encoder", config)
        pretrained_name = model_cfg.get("pretrained_name", "seyonec/ChemBERTa-zinc-base-v1")
        self.max_length = model_cfg.get("max_length", 512)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def encode(self, smiles: str) -> Dict[str, Any]:
        return self.tokenizer(smiles, max_length=self.max_length, truncation=True)

    def encode_batch(self, smiles_list: Iterable[str]) -> Dict[str, Any]:
        return self.tokenizer(list(smiles_list), max_length=self.max_length, truncation=True)
