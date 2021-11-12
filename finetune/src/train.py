import argparse

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import FineTuningDataModule, FineTuningModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"


def main(config: DictConfig):
    model_name = f"{config.train.name}-fold{config.data.fold_index}"
    model_checkpoint = ModelCheckpoint(monitor="val/score", save_weights_only=True)

    Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(project="mot-finetuning", name=model_name),
        callbacks=[model_checkpoint, LearningRateMonitor("step")],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        progress_bar_refresh_rate=1,
        log_every_n_steps=10,
    ).fit(FineTuningModule(config), datamodule=FineTuningDataModule(config))

    model = FineTuningModule.load_from_checkpoint(
        model_checkpoint.best_model_path, config=config
    )
    torch.save(model.state_dict(), model_name + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
