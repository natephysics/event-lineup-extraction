import pandas as pd
from msilib.schema import Component
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datamodules.components.event_data import EventDataSequence
from torch import seed
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Optional
from os import environ, path

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
        self.data_train = None,
        self.data_val = None,
        self.data_test = None,

        # set the seed for random numbers to the system default, otherwise 42
        try: 
            if environ["PL_GLOBAL_SEED"]:
                self.seed = int(environ["PL_GLOBAL_SEED"])
        except (TypeError, ValueError):
            self.seed = 42

    def setup(self, stage: Optional[str] = None) -> None:

        data_path = path.join(self.data_dir, "train_data.csv")
        data = pd.read_csv(data_path, sep='\t', dtype = str)
        # randomize the data
        data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # split the data into train, val and test
        self.data_train = EventDataSequence(data)
        self.data_train, self.data_val = self.data_train.split(self.train_val_split)
        self.data_train, self.data_test = self.data_train.split(self.train_val_split)
        return super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers)
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_test,
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
