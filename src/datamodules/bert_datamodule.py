from msilib.schema import Component
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datamodules.components import event_data
from numpy import split
from torch import seed
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Optional

class BertDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir, 
        train_val_split, 
        batch_size, 
        num_workers, 
        pin_memory,
        transforms=None,
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.transforms = transforms
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.data_dir = data_dir
        self.seed = 42
        self.data_train_data = None,
        self.data_val_data = None,
        self.data_test_data = None,

    def setup(self, stage: Optional[str] = None) -> None:
        if stage() == "fit":
            data = self.prepare_data()
            data_train, data_val = split(data.sample(frac=1, random_state=self.seed), [int(len(data) * self.train_val_split)])
            self.data_train_data = event_data(data_train)
            self.data_val_data = event_data(data_val)

        if stage() == "test":
            data = self.prepare_data()
            self.data_test_data = event_data(data.sample(frac=1, random_state=self.seed))
        return super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.data_train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers)
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers)
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"batch_size={self.batch_size}, "
            + f"pin_memory={self.pin_memory}, "
            + f"num_workers={self.num_workers}"
        )