from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from lipreading.datamodules.components.grid import GRIDDataset


class GRIDDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        transform,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = GRIDDataset(
                mode="train",
                dataset_cfg=self.hparams.dataset,
                transform_cfg=self.hparams.transform,
            )
            self.data_val = GRIDDataset(
                mode="val",
                dataset_cfg=self.hparams.dataset,
                transform_cfg=self.hparams.transform,
            )
        if stage == "validate":
            self.data_val = GRIDDataset(
                mode="val",
                dataset_cfg=self.hparams.dataset,
                transform_cfg=self.hparams.transform,
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = GRIDDataset(
                mode="test",
                dataset_cfg=self.hparams.dataset,
                transform_cfg=self.hparams.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from tqdm import tqdm

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "sentence" / "datamodule" / "grid.yaml"
    )
    cfg.dataset.data_dir = str(root / "data")
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    for i in tqdm(loader):
        pass
