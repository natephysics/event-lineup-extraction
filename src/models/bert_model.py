from transformers import BertForTokenClassification
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from typing import Optional, List
from src.utils import instantiate_list
import torch
import hydra
import pandas as pd

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
        self.lr = optimizer.lr if hasattr(optimizer, "lr") else None
        self.scheduler = scheduler
        self.log_training = log_training

        #  Whether to log gradient and weights during training
        self.log_training = log_training        

        # scheduler
        self.scheduler = scheduler

        # create a df for doing some post analysis
        self.test_results = pd.DataFrame(columns=['input_ids', 'label', 'predictions', 'attention_mask', 'token_type_ids'])

    def forward(self, input_id, mask, label):
        """Forward pass of the model."""
        loss, logits  = self.bert(input_ids=input_id[:,0], attention_mask=mask[:,0], labels=label[:,0], return_dict=False)

        return logits, loss

    def step(self, batch: any, batch_idx: int):

        # forward pass
        logits , loss = self.forward(batch["input_ids"], batch["attention_mask"], batch["label"])
        
        # convert the logits to predictions
        predictions = logits.argmax(dim=2)[:,None,:]

        # We only want predictions for the values not = -100
        predictions_clean = predictions[batch["label"] != -100]

        # We only want labels for the values not = -100
        label_clean = batch["label"][batch["label"] != -100]

        acc = (predictions_clean == label_clean).float().mean()

        if torch.isnan(loss):
            print('Issue')
        return loss, acc, logits, predictions

    def training_step(self, batch, batch_idx):
        loss, acc, logits, predictions = self.step(batch, batch_idx)
        
        # log the loss and accuracy
        self.log("train/loss", loss)
        self.log("train/acc", acc, prog_bar=True, logger=True)
        
        # log training metrics (if enabled)
        if self.log_training:
            self.train_metrics(predictions, batch['label'])
            self.log_dict(self.train_metrics)
            for layer, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    f"train/{layer}", param, global_step=self.global_step
                )
                if batch_idx != 0:
                    self.log(f"train/{layer}.max_grad", torch.max(param.grad))

        return {"loss": loss, "predictions": predictions, "labels": batch['label']}

    def validation_step(self, batch: any, batch_idx: int):
        loss, acc, logits, predictions = self.step(batch, batch_idx)
        
        # log the loss and accuracy
        self.log("val/loss", loss, prog_bar=True, logger=True)
        self.log("val/acc", acc, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: any, batch_idx: int):
        loss, acc, logits, predictions = self.step(batch, batch_idx)


        results = {**batch, "predictions": logits}

        for key, _ in results.items():
            results[key] = results[key].cpu().numpy()
            results[key] = [row for row in results[key]]

        # self.test_results
        self.test_results = self.test_results.append(pd.DataFrame.from_dict(results), ignore_index=True)

        # log the loss and accuracy
        self.log("test/loss", loss, prog_bar=True, logger=True)
        self.log("test/acc", acc, prog_bar=True, logger=True)
        
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

    # def compute_metrics(self, pred_label, true_label):
        
    #     # log training metrics (if enabled)
    #     if self.log_training:
    #         self.train_metrics(pred_label, true_label)
    #         self.log_dict(self.train_metrics)
    #         for layer, param in self.named_parameters():
    #             self.logger.experiment.add_histogram(
    #                 f"train/{layer}", param, global_step=self.global_step
    #             )
    #             if batch_idx != 0:
    #                 self.log(f"train/{layer}.max_grad", torch.max(param.grad))
        
        
        
    #     return self.test_metrics(pred_label, true_label)