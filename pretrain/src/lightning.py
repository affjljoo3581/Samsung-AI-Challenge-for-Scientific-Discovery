import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import BucketBatchSampler, MolecularParser, TabularDataset
from encoding import MolecularEncoder
from modeling import MoTConfig, MoTLayerNorm, MoTModel

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class PreTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        mot_config = MoTConfig(**MolecularEncoder.mot_config, **config.model.config)

        self.model = MoTModel(mot_config)
        self.classifier = nn.Linear(
            mot_config.hidden_dim,
            len(config.data.label_columns),
        )
        self.model.init_weights(self.classifier)

        if self.config.model.pretrained_model_path is not None:
            state_dict = torch.load(self.config.model.pretrained_model_path)
            self.model.load_state_dict(state_dict)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["attention_type_ids"],
            batch["position_ids"],
        )
        logits = self.classifier(hidden_states[:, 0, :])

        mse_loss = F.mse_loss(logits, batch["labels"].type_as(logits))
        mae_loss = F.l1_loss(logits, batch["labels"].type_as(logits))
        return mse_loss, mae_loss

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        mse_loss, mae_loss = self(batch)
        self.log("train/mse_loss", mse_loss)
        self.log("train/mae_loss", mae_loss)
        return mse_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], idx: int):
        mse_loss, mae_loss = self(batch)
        self.log("val/mse_loss", mse_loss)
        self.log("val/mae_loss", mae_loss)

    def create_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for the optimizer.

        Transformer-based models are usually optimized by AdamW (weight-decay decoupled
        Adam optimizer). And weight decaying are applied to only weight parameters, not
        bias and layernorm parameters. Hence, this method creates parameter groups which
        contain parameters for weight-decay and ones for non-weight-decay. Using this
        paramter groups, you can separate which parameters should not be decayed from
        entire parameters in this model.

        Returns:
            A list of parameter groups.
        """
        do_decay_params, no_decay_params = [], []
        for layer in self.modules():
            for name, param in layer.named_parameters(recurse=False):
                if isinstance(layer, MoTLayerNorm) or name == "bias":
                    no_decay_params.append(param)
                else:
                    do_decay_params.append(param)

        return [
            {"params": do_decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def adjust_learning_rate(self, current_step: int) -> float:
        """Calculate a learning rate scale corresponding to current step.

        MoT pretraining uses a linear learning rate decay with warmups. This method
        calculates the learning rate scale according to the linear warmup decaying
        schedule. Using this method, you can create a learning rate scheduler through
        `LambdaLR`.

        Args:
            current_step: A current step of training.

        Returns:
            A learning rate scale corresponding to current step.
        """
        if current_step < self.config.train.warmup_steps:
            return current_step / self.config.train.warmup_steps

        current_step = current_step - self.config.train.warmup_steps
        total_steps = self.config.train.training_steps - self.config.train.warmup_steps
        return max(0, 1 - current_step / total_steps)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(self.create_param_groups(), **self.config.train.optimizer)
        scheduler = LambdaLR(optimizer, self.adjust_learning_rate)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        with open(self.config.data.dataset_file.index, "rb") as fp:
            seeking_points, sequence_lengths = pickle.load(fp)

        num_validations = int(len(seeking_points) * self.config.data.validation_ratio)
        self.encoder = MolecularEncoder()

        # Create train and validation datasets by using splitted seeking point list. The
        # arguments are too long, so we pull the same arguments to outside to remove
        # duplications.
        dataset_kwargs = dict(
            filename=self.config.data.dataset_file.label,
            input_column=self.config.data.input_column,
            label_columns=self.config.data.label_columns,
            labels_mean_std=self.config.data.labels_mean_std,
        )
        self.train_dataset = TabularDataset(
            **dataset_kwargs,
            input_parser=MolecularParser(self.encoder, self.config.data.bond_drop_prob),
            seeking_points=seeking_points[num_validations:]
        )
        self.val_dataset = TabularDataset(
            **dataset_kwargs,
            input_parser=MolecularParser(self.encoder, bond_drop_prob=0.0),
            seeking_points=seeking_points[:num_validations]
        )

        self.train_batch_sampler = BucketBatchSampler(
            sequence_lengths[num_validations:], batch_size=self.config.train.batch_size
        )
        self.val_batch_sampler = BucketBatchSampler(
            sequence_lengths[:num_validations], batch_size=self.config.train.batch_size
        )

    @property
    def num_dataloader_workers(self) -> int:
        """Get the number of parallel workers in each dataloader."""
        if self.config.data.dataloader_workers >= 0:
            return self.config.data.dataloader_workers
        return os.cpu_count()

    def dataloader_collate_fn(self, features: List) -> Dict[str, torch.Tensor]:
        """Simple datacollate binder for dataloaders."""
        return self.encoder.collate(
            features,
            max_length=self.config.data.max_length,
            pad_to_multiple_of=8,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.dataloader_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.dataloader_collate_fn,
            persistent_workers=True,
        )
