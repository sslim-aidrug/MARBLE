"""Protein BERT student encoder component."""

from typing import Any, Dict, Iterable, Optional

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from components.base import BaseProteinEncoder


class ProteinEncoder(BaseProteinEncoder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        tokenizer_name = config.get("tokenizer_name", "Rostlab/prot_bert_bfd")
        max_length = config.get("max_length", 545)
        hidden_size = config.get("hidden_size", 512)
        num_hidden_layers = config.get("num_hidden_layers", 4)
        num_attention_heads = config.get("num_attention_heads", 4)
        intermediate_size = config.get("intermediate_size", 2048)
        hidden_act = config.get("hidden_act", "gelu")
        pad_token_id = config.get("pad_token_id", 0)

        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False
        )

        bert_config = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_length + 2,
            type_vocab_size=1,
            pad_token_id=pad_token_id,
            position_embedding_type="absolute",
        )

        self.encoder = BertModel(bert_config)
        self.output_dim = int(hidden_size)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def forward(self, inputs: Any) -> torch.Tensor:
        return self.encoder(**inputs).last_hidden_state[:, 0]


class ProteinDataLoader:
    """Tokenizer wrapper for registry-based data loading."""

    def __init__(self, config: Dict[str, Any], base_path: Optional[str] = None):
        _ = base_path
        self.config = config
        model_cfg = config.get("model", {}).get("protein_encoder", config)
        tokenizer_name = model_cfg.get("tokenizer_name", "Rostlab/prot_bert_bfd")
        self.max_length = model_cfg.get("max_length", 545)
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False
        )

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def encode(self, fasta: str) -> Dict[str, Any]:
        return self.tokenizer(
            " ".join(fasta), max_length=self.max_length + 2, truncation=True
        )

    def encode_batch(self, fasta_list: Iterable[str]) -> Dict[str, Any]:
        sequences = [" ".join(fasta) for fasta in fasta_list]
        return self.tokenizer(
            sequences, max_length=self.max_length + 2, truncation=True
        )
