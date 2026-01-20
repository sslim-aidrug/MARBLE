"""DLM-DTI Protein Encoder Template

This file is a template for LLM (code expert) to implement code based on papers.

[LLM CODING RULES]
1. Class name MUST remain 'ProteinEncoder' - DO NOT CHANGE
2. MUST inherit from BaseProteinEncoder
3. MUST implement: __init__, get_output_dim, get_tokenizer, forward methods
4. forward input: tokenized FASTA batch (dict of tensors)
5. forward output: torch.Tensor - shape (batch_size, output_dim)
6. Read architecture parameters from config

[Config Example - model.protein_encoder section in config.yaml]
protein_encoder:
  type: protein_encoder_other
  input:
    tokenizer_name: Rostlab/prot_bert_bfd
    max_length: 545
    add_spaces: true
  architecture:
    hidden_dim: 1024
    # ... additional parameters based on paper
"""

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from components.base import BaseProteinEncoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define helper classes here BEFORE the main encoder class.




# ==============================================================================
# MAIN PROTEIN ENCODER CLASS
# ==============================================================================
class ProteinEncoder(BaseProteinEncoder):
    """Protein Encoder for DLM-DTI

    [LLM IMPLEMENTATION AREA]
    - Implement the protein encoder architecture proposed in the paper
    - MUST define self.output_dim
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        input_cfg = config.get("input", {})
        arch_cfg = config.get("architecture", {})

        self.tokenizer_name = input_cfg.get("tokenizer_name", "Rostlab/prot_bert_bfd")
        self.max_length = input_cfg.get("max_length", 545)
        self.add_spaces = input_cfg.get("add_spaces", True)
        self.hidden_dim = arch_cfg.get("hidden_dim", 1024)

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Define self.output_dim
        # 2. Define encoder layers (Transformers, CNN, etc.)
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_tokenizer(self) -> Any:
        """Return tokenizer for protein sequences (used by data loader)."""
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: Tokenized protein batch (dict of tensors)

        Returns:
            Protein embedding (batch_size, output_dim)
        """
        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")


# ==============================================================================
# PROTEIN DATA LOADER - DO NOT MODIFY
# ==============================================================================
class ProteinDataLoader:
    """Tokenizer wrapper for registry-based data loading - DO NOT MODIFY."""

    def __init__(self, config: Dict[str, Any], base_path: Optional[str] = None):
        _ = base_path
        self.config = config
        encoder_cfg = config.get("model", {}).get("protein_encoder", config)
        input_cfg = encoder_cfg.get("input", encoder_cfg)
        tokenizer_name = input_cfg.get("tokenizer_name", "Rostlab/prot_bert_bfd")
        self.max_length = input_cfg.get("max_length", 545)
        self.add_spaces = input_cfg.get("add_spaces", True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def _prep(self, fasta: str) -> str:
        return " ".join(fasta) if self.add_spaces else fasta

    def encode(self, fasta: str) -> Dict[str, Any]:
        return self.tokenizer(
            self._prep(fasta), max_length=self.max_length + 2, truncation=True
        )

    def encode_batch(self, fasta_list: Iterable[str]) -> Dict[str, Any]:
        sequences = [self._prep(fasta) for fasta in fasta_list]
        return self.tokenizer(
            sequences, max_length=self.max_length + 2, truncation=True
        )
