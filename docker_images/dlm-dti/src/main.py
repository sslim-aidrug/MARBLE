"""Refactored training entrypoint for DLM-DTI."""

import argparse
import os
import time
from typing import Any, Dict

import pytorch_lightning as pl
import yaml

from components import (
    get_decoder,
    get_drug_data_loader,
    get_drug_encoder,
    get_protein_data_loader,
    get_protein_encoder,
)
from src.model_interface import DTI_prediction, define_callbacks
from src.model import DLMModel
from src.utils.data import get_dataloaders, load_cached_prot_features, load_dataset
from src.utils.eval import evaluate
from src.utils.logging import logging


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def _get_default_config_path() -> str:
    """Get default config path relative to script location (supports build_N iterations)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    return os.path.join(parent_dir, "config.yaml")


def _resolve_config_path(config_path: str | None) -> str:
    if config_path:
        return config_path

    fallback = _get_default_config_path()
    if os.path.exists(fallback):
        return fallback

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    alt_paths = [
        os.path.join(parent_dir, "config.yaml"),
        "./config.yaml",
    ]
    for path in alt_paths:
        if os.path.exists(path):
            return path

    return fallback


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "model" in cfg and "data" in cfg:
        return cfg

    dataset = cfg.get("dataset", "DAVIS")
    prot_length = cfg.get("prot_length", {"teacher": 545, "student": 545})
    prot_cfg = cfg.get("prot_encoder", {})
    train_cfg = cfg.get("training_config", {})
    lambda_cfg = cfg.get("lambda", {})

    return {
        "data": {
            "dataset": dataset,
            "prot_length": prot_length,
        },
        "model": {
            "drug_encoder": {
                "type": "drug_encoder_dlm",
                "pretrained_name": "seyonec/ChemBERTa-zinc-base-v1",
                "freeze_embeddings": True,
                "freeze_layers": 6,
            },
            "protein_encoder": {
                "type": "protein_encoder_dlm",
                "tokenizer_name": "Rostlab/prot_bert_bfd",
                "max_length": prot_length.get("student", 545),
                "hidden_size": prot_cfg.get("hidden_size", 512),
                "num_hidden_layers": prot_cfg.get("num_hidden_layers", 4),
                "num_attention_heads": prot_cfg.get("num_attention_heads", 4),
                "intermediate_size": prot_cfg.get("intermediate_size", 2048),
                "hidden_act": prot_cfg.get("hidden_act", "gelu"),
            },
            "decoder": {
                "type": "decoder_dlm",
                "hidden_dim": train_cfg.get("hidden_dim", 512),
                "teacher_dim": 1024,
                "dropout": 0.1,
                "learnable_lambda": lambda_cfg.get("learnable", True),
                "fixed_lambda": lambda_cfg.get("fixed_value", -1),
            },
        },
        "training": {
            "device": cfg.get("device", 0),
            "batch_size": train_cfg.get("batch_size", 32),
            "num_workers": train_cfg.get("num_workers", 0),
            "epochs": train_cfg.get("epochs", 50),
            "learning_rate": train_cfg.get("learning_rate", 0.0001),
            "precision": 16,
        },
    }


def _build_project_name(cfg: Dict[str, Any]) -> str:
    dataset = cfg["data"]["dataset"]
    teacher_len = cfg["data"]["prot_length"]["teacher"]
    student_len = cfg["data"]["prot_length"]["student"]

    decoder_cfg = cfg["model"]["decoder"]
    if decoder_cfg.get("learnable_lambda", True):
        lambda_status = "learnable"
    else:
        lambda_status = f"fixed-{decoder_cfg.get('fixed_lambda', -1)}"

    return (
        f"Dataset-{dataset}_ProtT-{teacher_len}_ProtS-{student_len}_Lambda-{lambda_status}"
    )


def _safe_avg(values: list[float]) -> float:
    valid = [v for v in values if v is not None and v >= 0]
    if not valid:
        return -1.0
    return float(sum(valid) / len(valid))


def _format_results(results: list[float], dataset: str) -> Dict[str, float]:
    keys = [
        "davis_auroc",
        "davis_auprc",
        "davis_sensitivity",
        "davis_specificity",
        "binding_auroc",
        "binding_auprc",
        "binding_sensitivity",
        "binding_specificity",
        "biosnap_auroc",
        "biosnap_auprc",
        "biosnap_sensitivity",
        "biosnap_specificity",
    ]
    results_dict = {k: float(v) for k, v in zip(keys, results)}

    dataset_key = (dataset or "").lower()
    prefix_map = {
        "davis": "davis",
        "bindingdb": "binding",
        "biosnap": "biosnap",
        "drugban": "davis",
    }
    if dataset_key == "merged":
        results_dict["auroc"] = _safe_avg(
            [
                results_dict.get("davis_auroc", -1),
                results_dict.get("binding_auroc", -1),
                results_dict.get("biosnap_auroc", -1),
            ]
        )
        results_dict["auprc"] = _safe_avg(
            [
                results_dict.get("davis_auprc", -1),
                results_dict.get("binding_auprc", -1),
                results_dict.get("biosnap_auprc", -1),
            ]
        )
    elif dataset_key in prefix_map:
        prefix = prefix_map[dataset_key]
        results_dict["auroc"] = results_dict.get(f"{prefix}_auroc", -1.0)
        results_dict["auprc"] = results_dict.get(f"{prefix}_auprc", -1.0)

    return results_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Dual Language Model based Drug-Target Interactions Prediction"
    )
    parser.add_argument("--config", type=str, default=None, help="specify config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg = _normalize_config(load_config(config_path))

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    cfg["training"]["epochs"] = 50

    project_name = _build_project_name(cfg)
    print(f"\nProject name: {project_name}\n")

    dataset = cfg["data"]["dataset"]
    display_dataset = "Davis" if dataset.lower() == "drugban" else dataset
    base_path = cfg["data"].get("base_path")
    train_df, valid_df, test_df = load_dataset(mode=dataset, base_path=base_path)
    print(f"Load Dataset: {display_dataset}")

    prot_feat_teacher = load_cached_prot_features(
        max_length=cfg["data"]["prot_length"]["teacher"]
    )
    print(
        f"Load Prot teacher's features; Prot Length {cfg['data']['prot_length']['teacher']}"
    )

    drug_encoder_cfg = cfg["model"]["drug_encoder"]
    protein_encoder_cfg = cfg["model"]["protein_encoder"]
    decoder_cfg = cfg["model"]["decoder"]

    drug_encoder_cls = get_drug_encoder(drug_encoder_cfg["type"])
    protein_encoder_cls = get_protein_encoder(protein_encoder_cfg["type"])
    decoder_cls = get_decoder(decoder_cfg["type"])

    drug_encoder = drug_encoder_cls(drug_encoder_cfg)
    protein_encoder = protein_encoder_cls(protein_encoder_cfg)

    drug_data_loader_cls = get_drug_data_loader(drug_encoder_cfg["type"])
    protein_data_loader_cls = get_protein_data_loader(protein_encoder_cfg["type"])
    drug_data_loader = drug_data_loader_cls(cfg)
    protein_data_loader = protein_data_loader_cls(cfg)

    mol_tokenizer = drug_data_loader.get_tokenizer()
    prot_tokenizer = protein_data_loader.get_tokenizer()

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        train_df,
        valid_df,
        test_df,
        prot_feat_teacher=prot_feat_teacher,
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        max_lenght=cfg["data"]["prot_length"]["student"],
        d_mode=dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    decoder = decoder_cls(
        decoder_cfg,
        drug_dim=drug_encoder.get_output_dim(),
        protein_dim=protein_encoder.get_output_dim(),
    )

    model = DLMModel(drug_encoder, protein_encoder, decoder)
    callbacks = define_callbacks(project_name)

    model_interface = DTI_prediction(
        model, len(train_dataloader), cfg["training"]["learning_rate"]
    )

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        gpus=[cfg["training"]["device"]],
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=20,
        precision=cfg["training"].get("precision", 16),
        logger=False,
    )

    start_time = time.time()
    trainer.fit(model_interface, train_dataloader, valid_dataloader)
    end_time = time.time()

    predictions = trainer.predict(model_interface, test_dataloader, ckpt_path="best")
    results = evaluate(predictions)
    results_dict = _format_results(results, dataset)
    print(f"Results: {results_dict}", flush=True)
    logging(project_name, end_time - start_time, results)


if __name__ == "__main__":
    main()
