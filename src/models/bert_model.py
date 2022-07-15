from transformers import BertForTokenClassification
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from typing import Optional, List
from src.utils import instantiate_list
import torch

class BertModel(pl.LightningModule):

    def __init__(
        self, 
        num_labels: int, 
        log_training: Optional[bool] = False,
        optimizer: Optional[DictConfig] = {},
        scheduler: Optional[DictConfig] = {},
        metrics: Optional[DictConfig] = {},
        ) -> None:

        super(BertModel, self).__init__()
        self.save_hyperparameters(
            "optimizer", "scheduler", "metrics"
        )

        # load the model
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)

        # metrics
        metrics = instantiate_list(metrics)
        metrics = MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # define "lr" as attribute to leverage pl learning rate tuner
        self.optimizer = optimizer
        self.lr = optimizer.lr

        #  Whether to log gradient and weights during training
        self.log_training = log_training        

        # scheduler
        self.scheduler = scheduler

    def forward(self, input_id, mask, label):
        """Forward pass of the model."""
        loss, output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output, loss

    def step(self, batch: any, batch_idx: int):
        # get the inputs
        data, true_label = batch
        # forward pass
        output, loss = self.forward(data["input_ids"][0], data["attention_mask"][0], data["labels"][0])
        # compute the loss
        output_clean = output[0][output != -100]
        label_clean = true_label[true_label != -100]

        predictions = output_clean.argmax(dim=1)

        acc = (predictions == label_clean).float().mean()
        self.log("train/acc", acc)


        if torch.isnan(loss):
            print('Issue')
        return loss, predictions, label_clean

    def training_step(self, batch, batch_idx):
        loss, pred_label, true_label = self.step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True)

        # log training metrics (if enabled)
        if self.log_training:
            self.train_metrics(pred_label, true_label)
            self.log_dict(self.train_metrics)
            for layer, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    f"train/{layer}", param, global_step=self.global_step
                )
                if batch_idx != 0:
                    self.log(f"train/{layer}.max_grad", torch.max(param.grad))

        return {"loss": loss, "predictions": pred_label, "labels": true_label}

    def validation_step(self, batch: any, batch_idx: int):
        loss, pred_label, true_label = self.step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: any, batch_idx: int):
        loss, pred_label, true_label = self.step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        if self.scheduler != {}:
            scheduler = self.scheduler.copy()
            monitor = scheduler.pop("monitor", None)
            lr_scheduler = {
                "scheduler": instantiate(scheduler, optimizer),
                "monitor": monitor,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer
