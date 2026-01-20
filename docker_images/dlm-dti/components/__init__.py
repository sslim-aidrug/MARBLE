"""DLM-DTI component registry exports."""

from components.registry import (
    get_decoder,
    get_drug_data_loader,
    get_drug_encoder,
    get_protein_data_loader,
    get_protein_encoder,
    is_registered,
    list_available_components,
    register_decoder,
    register_drug_data_loader,
    register_drug_encoder,
    register_protein_data_loader,
    register_protein_encoder,
)

__all__ = [
    "get_decoder",
    "get_drug_data_loader",
    "get_drug_encoder",
    "get_protein_data_loader",
    "get_protein_encoder",
    "is_registered",
    "list_available_components",
    "register_decoder",
    "register_drug_data_loader",
    "register_drug_encoder",
    "register_protein_data_loader",
    "register_protein_encoder",
]
