import argparse
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn as nn

from lightning import PreTrainingDataModule, PreTrainingModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    if resume_from and config.train.gpus == 1 and amp_backend == "apex":
        # Since there is a bug when we resume training with `apex` backend, we resolve
        # that problem by patching `apex` with dummy module.
        apex.amp.initialize(nn.Linear(1, 1).cuda())

    model_checkpoint = ModelCheckpoint(save_last=True)
    Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(
            project="mot-pretraining", name=config.train.name, id=resume_id
        ),
        callbacks=[model_checkpoint, LearningRateMonitor("step")],
        precision=config.train.precision,
        max_steps=config.train.training_steps,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=config.train.validation_interval,
        resume_from_checkpoint=resume_from,
        accumulate_grad_batches=config.train.accumulate_grads,
    ).fit(PreTrainingModule(config), datamodule=PreTrainingDataModule(config))

    # Save the pretrained model weights by loading the saved checkpoint and extracting
    # MoT model weights.
    model = PreTrainingModule.load_from_checkpoint(
        model_checkpoint.last_model_path, config=config
    )
    torch.save(model.model.state_dict(), config.train.name + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config, args.resume_from, args.resume_id)
