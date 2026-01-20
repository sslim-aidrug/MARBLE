"""Hint-based MLP decoder with learnable lambda mixing."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.base import BaseDecoder


class Decoder(BaseDecoder):
    def __init__(self, config: Dict, drug_dim: int, protein_dim: int):
        super().__init__(config, drug_dim, protein_dim)

        hidden_dim = config.get("hidden_dim", 512)
        teacher_dim = config.get("teacher_dim", 1024)
        dropout = config.get("dropout", 0.1)
        learnable_lambda = config.get("learnable_lambda", True)
        fixed_lambda = config.get("fixed_lambda", -1)

        self.hidden_dim = hidden_dim
        self.teacher_dim = teacher_dim
        self.dropout = dropout
        self.is_learnable_lambda = learnable_lambda

        if self.is_learnable_lambda:
            self.lambda_param = nn.Parameter(torch.rand(1))
        else:
            if fixed_lambda < 0 or fixed_lambda > 1:
                raise ValueError("fixed_lambda must be between 0 and 1 when learnable_lambda is False")
            self.register_buffer("lambda_param", torch.tensor(float(fixed_lambda)))

        self.molecule_align = nn.Sequential(
            nn.LayerNorm(drug_dim), nn.Linear(drug_dim, hidden_dim, bias=False)
        )

        self.protein_align_teacher = nn.Sequential(
            nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, hidden_dim, bias=False)
        )

        self.protein_align_student = nn.Sequential(
            nn.LayerNorm(protein_dim), nn.Linear(protein_dim, hidden_dim, bias=False)
        )

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cls_out = nn.Linear(hidden_dim, 1)

    def get_output_dim(self) -> int:
        return 1

    def forward(self, mol_feat, prot_feat_student, prot_feat_teacher=None):
        if prot_feat_teacher is None:
            raise ValueError("prot_feat_teacher is required for decoder_hint_mlp")

        mol_feat = self.molecule_align(mol_feat)
        prot_feat_student = self.protein_align_student(prot_feat_student)
        prot_feat_teacher = self.protein_align_teacher(prot_feat_teacher).squeeze(1)

        if self.is_learnable_lambda:
            lambda_val = torch.sigmoid(self.lambda_param)
        else:
            lambda_val = self.lambda_param.detach()

        merged_prot_feat = lambda_val * prot_feat_student + (1 - lambda_val) * prot_feat_teacher
        x = torch.cat([mol_feat, merged_prot_feat], dim=1)

        x = F.dropout(F.gelu(self.fc1(x)), self.dropout)
        x = F.dropout(F.gelu(self.fc2(x)), self.dropout)
        x = F.dropout(F.gelu(self.fc3(x)), self.dropout)

        logits = self.cls_out(x).squeeze(-1)
        return logits, lambda_val.mean()
