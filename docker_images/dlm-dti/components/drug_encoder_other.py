"""DLM-DTI Drug Encoder Template

This file is a template for LLM (code expert) to implement code based on papers.

[LLM CODING RULES]
1. Class name MUST remain 'DrugEncoder' - DO NOT CHANGE
2. MUST inherit from BaseDrugEncoder
3. MUST implement: __init__, get_output_dim, get_tokenizer, forward methods
4. forward input: tokenized SMILES batch (dict of tensors)
5. forward output: torch.Tensor - shape (batch_size, output_dim)
6. Read architecture parameters from config

[Config Example - model.drug_encoder section in config.yaml]
drug_encoder:
  type: drug_encoder_other
  input:
    tokenizer_name: seyonec/ChemBERTa-zinc-base-v1
    max_length: 512
  architecture:
    hidden_dim: 768
    # ... additional parameters based on paper
"""

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from components.base import BaseDrugEncoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define helper classes here BEFORE the main encoder class.




# ==============================================================================
# MAIN DRUG ENCODER CLASS
# ==============================================================================
class DrugEncoder(BaseDrugEncoder):
    """Drug Encoder for DLM-DTI

    [LLM IMPLEMENTATION AREA]
    - Implement the drug encoder architecture proposed in the paper
    - MUST define self.output_dim
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        input_cfg = config.get("input", {})
        arch_cfg = config.get("architecture", {})

        self.tokenizer_name = input_cfg.get(
            "tokenizer_name", "seyonec/ChemBERTa-zinc-base-v1"
        )
        self.max_length = input_cfg.get("max_length", 512)
        self.hidden_dim = arch_cfg.get("hidden_dim", 768)

        # ============================================================
        # [LLM IMPLEMENTATION AREA]
        # 1. Define self.output_dim
        # 2. Define encoder layers (Transformers, CNN, GNN, etc.)
        # ============================================================

        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_tokenizer(self) -> Any:
        """Return tokenizer for SMILES (used by data loader)."""
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: Tokenized SMILES batch (dict of tensors)

        Returns:
            Drug embedding (batch_size, output_dim)
        """
        # TODO: LLM implements based on paper
        raise NotImplementedError("LLM must implement this section")


# ==============================================================================
# DRUG DATA LOADER - DO NOT MODIFY
# ==============================================================================
class DrugDataLoader:
    """Tokenizer wrapper for registry-based data loading - DO NOT MODIFY."""

    def __init__(self, config: Dict[str, Any], base_path: Optional[str] = None):
        _ = base_path
        self.config = config
        encoder_cfg = config.get("model", {}).get("drug_encoder", config)
        input_cfg = encoder_cfg.get("input", encoder_cfg)
        tokenizer_name = input_cfg.get(
            "tokenizer_name", "seyonec/ChemBERTa-zinc-base-v1"
        )
        self.max_length = input_cfg.get("max_length", 512)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def encode(self, smiles: str) -> Dict[str, Any]:
        return self.tokenizer(smiles, max_length=self.max_length, truncation=True)

    def encode_batch(self, smiles_list: Iterable[str]) -> Dict[str, Any]:
        return self.tokenizer(
            list(smiles_list), max_length=self.max_length, truncation=True
        )
