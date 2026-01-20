"""Base classes for DLM-DTI components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseDrugEncoder(nn.Module, ABC):
    """Drug (molecule) encoder interface."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.output_dim: int = 0

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return tokenizer used by the encoder."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        """Encode molecule tokens to a fixed-size embedding."""
        raise NotImplementedError


class BaseProteinEncoder(nn.Module, ABC):
    """Protein encoder interface (student encoder)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.output_dim: int = 0

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output embedding dimension."""
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return tokenizer used by the encoder."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        """Encode protein tokens to a fixed-size embedding."""
        raise NotImplementedError


class BaseDecoder(nn.Module, ABC):
    """Decoder interface for combining drug/protein features."""

    def __init__(self, config: Dict[str, Any], drug_dim: int, protein_dim: int):
        super().__init__()
        self.config = config
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.output_dim: int = 1

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        mol_feat: torch.Tensor,
        prot_feat_student: torch.Tensor,
        prot_feat_teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and lambda value."""
        raise NotImplementedError
