"""DLM-DTI Decoder Template

This file is a template for LLM (code expert) to implement code based on papers.

[LLM CODING RULES]
1. Class name MUST remain 'Decoder' - DO NOT CHANGE
2. MUST inherit from BaseDecoder
3. MUST implement: __init__, get_output_dim, forward methods
4. forward takes: drug_encoded, protein_encoded, prot_feat_teacher tensors
5. forward output: Tuple(logits, lambda_val)
6. Read architecture parameters from config

[Config Example - model.decoder section in config.yaml]
decoder:
  type: decoder_other
  architecture:
    fusion_type: hint_mix  # Options: hint_mix, concat, attention, bilinear
    hidden_dim: 1024
    teacher_dim: 1024
    dropout: 0.1
    learnable_lambda: true
    fixed_lambda: -1
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from components.base import BaseDecoder


# ==============================================================================
# AUXILIARY MODULES
# ==============================================================================
# Define helper classes here BEFORE the main decoder class.
# Examples: attention modules, bilinear fusion layers, gating blocks, etc.




# ==============================================================================
# MAIN DECODER CLASS
# ==============================================================================
class Decoder(BaseDecoder):
    """Decoder for DLM-DTI

    [LLM IMPLEMENTATION AREA]
    - Implement the decoder architecture proposed in the paper
    - Fuse drug and protein representations (optionally with teacher hint)
    - Predict interaction logits

    Args:
        model_config: Full model configuration dictionary
        drug_dim: Output dimension from drug encoder
        protein_dim: Output dimension from protein encoder
    """

    def __init__(self, model_config: Dict[str, Any], drug_dim: int, protein_dim: int):
        super().__init__(model_config, drug_dim, protein_dim)

        # ============================================================
        # [FIXED] Read basic dimensions from config
        # ============================================================
        decoder_config = model_config.get("decoder", model_config)
        arch = decoder_config.get("architecture", decoder_config)

        self.hidden_dim = arch.get("hidden_dim", 1024)
        self.teacher_dim = arch.get("teacher_dim", 1024)
        self.dropout = arch.get("dropout", 0.1)
        self.output_dim = 1

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Read additional parameters
        # Example:
        # self.fusion_type = arch.get("fusion_type", "hint_mix")
        # self.learnable_lambda = arch.get("learnable_lambda", True)
        # self.fixed_lambda = arch.get("fixed_lambda", -1)
        # ============================================================

        # TODO: LLM implements parameter reading based on paper

        # ============================================================
        # [LLM IMPLEMENTATION AREA] Define layers
        # Example (hint-based mixing):
        # self.molecule_align = nn.Sequential(...)
        # self.protein_align_student = nn.Sequential(...)
        # self.protein_align_teacher = nn.Sequential(...)
        # self.fc1 = nn.Linear(...)
        # ============================================================

        # TODO: LLM implements layer definitions based on paper
        raise NotImplementedError("LLM must implement this section")

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        drug_encoded: torch.Tensor,
        protein_encoded: torch.Tensor,
        prot_feat_teacher: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        [FIXED] Input/Output format
        Args:
            drug_encoded: Drug embedding from drug encoder (batch_size, drug_dim)
            protein_encoded: Protein embedding from protein encoder (batch_size, protein_dim)
            prot_feat_teacher: Optional teacher features (batch_size, 1, teacher_dim)
            **kwargs: Additional arguments (ignored if not needed)

        Returns:
            logits: (batch_size,) binary logit output
            lambda_val: scalar tensor (mean lambda used for mixing)

        [LLM IMPLEMENTATION AREA]
        - Fuse drug/protein (and teacher) representations
        - Pass through classifier
        - Return (logits, lambda_val)
        """
        # ============================================================
        # [LLM IMPLEMENTATION AREA] Forward pass logic
        # Example:
        # fused = torch.cat([drug_encoded, protein_encoded], dim=1)
        # logits = self.classifier(fused).squeeze(-1)
        # lambda_val = torch.zeros(1, device=logits.device)
        # return logits, lambda_val
        # ============================================================

        # TODO: LLM implements forward pass based on paper
        raise NotImplementedError("LLM must implement this section")
