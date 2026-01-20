"""Refactored DLM-DTI model wiring."""

import torch
import torch.nn as nn


class DLMModel(nn.Module):
    def __init__(self, drug_encoder: nn.Module, protein_encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.decoder = decoder

    def forward(self, smiles_tokens, fasta_tokens, prot_feat_teacher):
        mol_feat = self.drug_encoder(smiles_tokens)
        prot_feat_student = self.protein_encoder(fasta_tokens)
        logits, lambda_val = self.decoder(
            mol_feat, prot_feat_student, prot_feat_teacher
        )
        return logits, lambda_val
