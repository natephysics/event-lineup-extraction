import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from regex import P
import torch
import pandas as pd
from src import utils

from rich.console import Console
from rich.table import Table

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, config.data_dir)
    datamodule.setup("test")

    # get the number of labels
    num_labels = len(datamodule.data_test.unique_labels)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        num_labels=num_labels,
        metrics=config.metrics,
        _recursive_=False, # for hydra (won't recursively instantiate criterion)
    )

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    model.test_results

    # Let's do some analysis on the test results

    def convert_to_tokens(inputs: List, tokenizer: torch.nn.DataParallel) -> List:
        return tokenizer.convert_ids_to_tokens(inputs)

    def convert_to_labels(label: List, label_map: dict) -> List:
        return [label_map[l] if l in label_map else None for l in label]


    def contract_labels(inputs: List, labels: List, predictions: List):
        # check that all three list have the same the same length:
        assert len(inputs) == len(labels) == len(predictions)   
        new_inputs = []
        new_labels = []
        new_predictions = []
        for input, label, prediction in zip(inputs, labels, predictions):
            if label is None:
                continue
            if input.startswith("##"):
                new_inputs[-1] += input[2:]
            else:
                new_inputs.append(input)
                new_labels.append(label)
                new_predictions.append(prediction)
        return new_inputs, new_labels, new_predictions

    model.test_results['inputs'] = model.test_results.input_ids.apply(
        lambda x: convert_to_tokens(x[0], datamodule.data_test.tokenizer)
        )

    model.test_results['label'] = model.test_results.label.apply(
        lambda x: convert_to_labels(x[0], datamodule.data_test.ids_to_labels)
        )

    model.test_results['predictions'] = model.test_results.predictions.apply(
        lambda x: convert_to_labels(x[0], datamodule.data_test.ids_to_labels)
        )

    temp = model.test_results.apply(lambda row: contract_labels(row.inputs, row.label, row.predictions), axis=1)

    model.test_results = pd.DataFrame(temp.tolist(), columns=['inputs', 'label', 'predictions'])

    def colorize(inputs: List, labels: List, predictions: List) -> List:
        """Colorize the input based on the results of the prediction."""
        assert len(inputs) == len(labels) == len(predictions)   
        color_inputs = []
        for input, label, prediction in zip(inputs, labels, predictions):
            if label == prediction:
                if label in ["B-per", "I-per"]:
                    color_inputs.append(f"[bold green]{input}[/bold green]")
                else:
                    color_inputs.append(input)
            else:
                if label in ["B-per", "I-per"]:
                    color_inputs.append(f"[bold red]{input}[/bold red]")
                else:
                    color_inputs.append(f"[red]{input}[/red]")
        return color_inputs

    # create colored inputs based on the results of the prediction
    model.test_results['color_inputs'] = model.test_results.apply(lambda row: colorize(row.inputs, row.label, row.predictions), axis=1)
    model.test_results['color_inputs'] = model.test_results['color_inputs'].apply(lambda x: ' '.join(x))

    # count the number of errors in each sentence and colorize the results
    model.test_results['errors'] = model.test_results.apply(lambda row: sum([label != predicion for label, predicion in zip(row.label, row.predictions)]), axis=1)
    model.test_results['errors'] = model.test_results['errors'].apply(lambda x: f"[green]0[/green]" if x == 0 else f"[red]{x}[/red]")

    # Create a table to make visual inspection of the results easier
    table = Table(title="Test data results")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Results", justify="left", no_wrap=True)
    table.add_column("# of Errors", justify="center")

    model.test_results.apply(lambda row: table.add_row(str(row.name), row.color_inputs, str(row.errors)), axis=1)
    console = Console()
    console.print(table)

    print('Done!')
