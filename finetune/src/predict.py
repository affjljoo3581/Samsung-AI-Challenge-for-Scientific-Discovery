import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import ST1_ENERGY_GAP_MEAN, ST1_ENERGY_GAP_STD, SSDDataset
from encoding import MolecularEncoder
from modeling import MoTConfig, MoTModel


class PredictionModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        mot_config = MoTConfig(**MolecularEncoder.mot_config, **config.model.config)

        self.model = MoTModel(mot_config)
        self.classifier = nn.Linear(mot_config.hidden_dim, 1)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["attention_type_ids"],
            batch["position_ids"],
        )
        return self.classifier(hidden_states[:, 0, :]).squeeze(-1)


def create_dataloader(config: DictConfig) -> DataLoader:
    # Read label csv files and collect SDF molfiles from the configuration. The
    # dataframes will be concatenated and used to find the labels when loading batches.
    datasets, structure_files = [], []
    for dataset in config.data.dataset_files:
        datasets.append(pd.read_csv(dataset["labels"], index_col="uid"))
        structure_files += [
            os.path.join(dataset["structures"], filename)
            for filename in os.listdir(dataset["structures"])
        ]

    dataset = pd.concat(datasets)
    encoder = MolecularEncoder()

    # Define a collate function to stack individual samples into the batch. This process
    # is necessary because each sample has different length of sequence. To run the
    # model in parallel, it is important to match the lengths.
    def collate_fn(features: List) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        uids = [uid for uid, encoding in features]
        encodings = [encoding for uid, encoding in features]
        return uids, encoder.collate(encodings, config.data.max_length, 8)

    return DataLoader(
        dataset=SSDDataset(dataset, structure_files, encoder, bond_drop_prob=0.0),
        batch_size=config.predict.batch_size,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def main(config: DictConfig):
    dataloader = create_dataloader(config)

    model = PredictionModel(config).eval().cuda()
    model.load_state_dict(torch.load(config.model.pretrained_model_path))

    preds = []
    for uids, batch in tqdm.tqdm(dataloader):
        batch = {
            "input_ids": [x.cuda() for x in batch["input_ids"]],
            "attention_mask": batch["attention_mask"].cuda(),
            "attention_type_ids": batch["attention_type_ids"].cuda(),
            "position_ids": batch["position_ids"].cuda(),
        }
        for uid, target in zip(uids, model(batch).tolist()):
            target = target * ST1_ENERGY_GAP_STD + ST1_ENERGY_GAP_MEAN
            preds.append({"uid": uid, "ST1_GAP(eV)": abs(target)})

    preds = pd.DataFrame(preds).set_index("uid")
    preds.to_csv(config.model.pretrained_model_path.replace(".pth", ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
