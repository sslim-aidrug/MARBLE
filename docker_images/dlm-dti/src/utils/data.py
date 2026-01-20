import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "Protein" in df.columns and "Target Sequence" not in df.columns:
        rename_map["Protein"] = "Target Sequence"
    if "Y" in df.columns and "Label" not in df.columns:
        rename_map["Y"] = "Label"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _read_split(train_path: str, valid_path: str, test_path: str):
    train_df = _standardize_columns(pd.read_csv(train_path))
    valid_df = _standardize_columns(pd.read_csv(valid_path))
    test_df = _standardize_columns(pd.read_csv(test_path))
    return train_df, valid_df, test_df


def load_dataset(mode="DAVIS", base_path=None):
    if base_path:
        train_path = os.path.join(base_path, "train.csv")
        valid_path = os.path.join(base_path, "val.csv")
        test_path = os.path.join(base_path, "test.csv")
        if all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
            return _read_split(train_path, valid_path, test_path)

    base_dir = base_path or "./data"

    if mode in ["DAVIS", "BindingDB", "BIOSNAP"]:
        train_path = os.path.join(base_dir, f"{mode}_train.csv")
        valid_path = os.path.join(base_dir, f"{mode}_valid.csv")
        test_path = os.path.join(base_dir, f"{mode}_test.csv")
        if not os.path.exists(train_path) and base_path:
            base_dir = "./data"
            train_path = os.path.join(base_dir, f"{mode}_train.csv")
            valid_path = os.path.join(base_dir, f"{mode}_valid.csv")
            test_path = os.path.join(base_dir, f"{mode}_test.csv")
        return _read_split(train_path, valid_path, test_path)
    elif mode == "merged":
        print("Load merged datasets")
        train_path = os.path.join(base_dir, "train_dataset.csv")
        valid_path = os.path.join(base_dir, "valid_dataset.csv")
        test_path = os.path.join(base_dir, "test_dataset.csv")
        if not os.path.exists(train_path) and base_path:
            base_dir = "./data"
            train_path = os.path.join(base_dir, "train_dataset.csv")
            valid_path = os.path.join(base_dir, "valid_dataset.csv")
            test_path = os.path.join(base_dir, "test_dataset.csv")
        return _read_split(train_path, valid_path, test_path)

    raise ValueError(f"Unknown dataset mode: {mode}")


def load_cached_prot_features(max_length=1024):
    with open(f"prot_feat/{max_length}_cls.pkl", "rb") as f:
        prot_feat_teacher = pickle.load(f)

    return prot_feat_teacher


class DTIDataset(Dataset):
    def __init__(
        self,
        data,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_length,
        d_mode="merged",
    ):
        self.data = data
        self.prot_feat_teacher = prot_feat_teacher
        self.max_length = max_length
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.d_mode = d_mode

    def get_mol_feat(self, smiles):
        return self.mol_tokenizer(smiles, max_length=512, truncation=True)

    def get_prot_feat_student(self, fasta):
        return self.prot_tokenizer(
            " ".join(fasta), max_length=self.max_length + 2, truncation=True
        )

    def get_prot_feat_teacher(self, fasta):
        return self.prot_feat_teacher[fasta[:20]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data.loc[index, "SMILES"]
        mol_feat = self.get_mol_feat(smiles)

        fasta = self.data.loc[index, "Target Sequence"]
        prot_feat_student = self.get_prot_feat_student(fasta)
        prot_feat_teacher = self.get_prot_feat_teacher(fasta)

        y = self.data.loc[index, "Label"]

        if self.d_mode == "merged":
            source = self.data.loc[index, "Source"]
            if source == "DAVIS":
                source = 1
            elif source == "BindingDB":
                source = 2
            elif source == "BIOSNAP":
                source = 3
        elif self.d_mode == "DAVIS":
            source = 1
        elif self.d_mode == "BindingDB":
            source = 2
        elif self.d_mode == "BIOSNAP":
            source = 3
        else:
            source = 1

        return mol_feat, prot_feat_student, prot_feat_teacher, y, source


class CollateBatch(object):
    def __init__(self, mol_tokenizer, prot_tokenizer):
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer

    def __call__(self, batch):
        mol_features, prot_feat_student, prot_feat_teacher, y, source = (
            [],
            [],
            [],
            [],
            [],
        )

        for mol_seq, prot_seq_student, prot_seq_teacher, y_, source_ in batch:
            mol_features.append(mol_seq)
            prot_feat_student.append(prot_seq_student)
            prot_feat_teacher.append(prot_seq_teacher.detach().cpu().numpy().tolist())
            y.append(y_)
            source.append(source_)

        mol_features = self.mol_tokenizer.pad(mol_features, return_tensors="pt")
        prot_feat_student = self.prot_tokenizer.pad(
            prot_feat_student, return_tensors="pt"
        )
        prot_feat_teacher = torch.tensor(prot_feat_teacher).float()
        y = torch.tensor(y).float()
        source = torch.tensor(source)

        return mol_features, prot_feat_student, prot_feat_teacher, y, source


def define_balanced_sampler(train_df, target_col_name="Label"):
    counts = np.bincount(train_df[target_col_name])
    labels_weights = 1.0 / counts
    weights = labels_weights[train_df[target_col_name]]
    sampler = WeightedRandomSampler(weights, len(weights))

    return sampler


def get_dataloaders(
    train_df,
    valid_df,
    test_df,
    prot_feat_teacher,
    mol_tokenizer,
    prot_tokenizer,
    max_lenght,
    d_mode="merged",
    target_col_name="Label",
    batch_size=128,
    num_workers=-1,
):
    train_dataset = DTIDataset(
        train_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )
    valid_dataset = DTIDataset(
        valid_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )
    test_dataset = DTIDataset(
        test_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )

    # sampler = define_balanced_sampler(train_df, target_col_name)
    collator = CollateBatch(mol_tokenizer, prot_tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collator,
    )
    #   sampler=sampler, collate_fn=collator)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    return train_dataloader, valid_dataloader, test_dataloader
