from argparse import ArgumentParser
from functools import partial

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy, auroc

import pytorch_lightning as pl


class LitLogisticRegression(pl.LightningModule):
    """
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool=True,
        learning_rate: float=1e-4,
        optimizer: Optimizer = Adam,
        loss: str='multinomial',
        auroc_avg: str='macro',
        l2_alpha: float=0.
    ) -> None:
        """
        """
        if loss not in {'bin_xent', 'multinomial'}:
            raise ValueError('[ERROR] only supports `bin_xent` and `multinomial`!')

        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self._init_model(auroc_avg)

    def _init_model(self, auroc_avg: str):
        """
        """
        self._model = nn.Linear(self.hparams.input_dim, self.hparams.num_classes,
                                bias=self.hparams.fit_intercept)

        if self.hparams.loss == 'bin_xent':
            self._nonlin = torch.sigmoid
            self._acc_metric = (
                lambda x, y:
                    auroc(torch.logsigmoid(x), y,
                          average=auroc_avg,
                          num_classes=self.hparams.num_classes)
            )
            self._loss_fn = (
                lambda x, y, reduction:
                    F.binary_cross_entropy_with_logits(
                        x, y, reduction=reduction
                    )
            )

        elif self.hparams.loss == 'multinomial':
            # Pytorch cross_entropy function combins log_softmax and nll_loss in single function
            self._nonlin = partial(F.softmax, dim=1)
            self._acc_metric = lambda x, y: accuracy(F.softmax(x, dim=1), y)
            self._loss_fn = (
                lambda x, y, reduction:
                    F.cross_entropy(x, y, reduction=reduction)
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        a = self._model(x)
        y = self._nonlin(a)
        return y

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        """
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self._model(x)
        loss = self._loss_fn(y_hat, y, 'sum')

        # L2 regularizer
        if self.hparams.l2_alpha > 0:
            l2_reg = self._model.weight.pow(2).sum()
            loss += self.hparams.l2_alpha * l2_reg

        loss /= x.size(0)

        tensorboard_logs = {'train_xe_loss': loss.detach()}
        progress_bar_metrics = tensorboard_logs
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self._model(x)
        acc = self._acc_metric(y_hat, y)
        return {"val_loss": self._loss_fn(y_hat, y, 'mean').detach(),
                "acc": acc, "n_samples": x.size(0)}

    def validation_epoch_end(
        self,
        outputs: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        """
        n_total_samples = sum(x['n_samples'] for x in outputs)
        acc = torch.stack([x["acc"] * x['n_samples'] for x in outputs]).sum() / n_total_samples
        val_loss = torch.stack([x["val_loss"] * x['n_samples'] for x in outputs]).sum() / n_total_samples
        tensorboard_logs = {"val_ce_loss": val_loss, "val_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"val_loss": val_loss.detach(), "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self._model(x)
        acc = self._acc_metric(y_hat, y)
        return {"test_loss": self._loss_fn(y_hat, y, 'mean'), "acc": acc, "n_samples": x.size(0)}

    def test_epoch_end(
        self,
        outputs: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        """
        n_total_samples = sum(x['n_samples'] for x in outputs)
        acc = torch.stack([x["acc"] * x['n_samples'] for x in outputs]).sum() / n_total_samples
        test_loss = torch.stack([x["test_loss"] * x['n_samples'] for x in outputs]).sum() / n_total_samples
        tensorboard_logs = {'test_xe_loss': test_loss, 'test_acc': acc}
        progress_bar_metrics = tensorboard_logs
        return {'test_loss': test_loss.detach(), 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--fit_intercept", default="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        return parser


class LitSKLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(
        self,
        alpha: float = 1e-4,
        learning_rate: float = 1e-4,
        max_iter: int = 1000,
        batch_size: int = 100,
        accelerator: str = 'cpu',
        loss: str = 'multinomial',
        num_workers: int = 1,
        verbose: bool = False
    ):
        """
        """
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.loss = loss
        self.num_workers = num_workers
        self.verbose = verbose

    def fit(self, X, y):
        """
        """
        # Check that X and y have correct shape
        if self.loss == 'multinomial':
            X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        X = torch.Tensor(X)
        if self.loss == 'multinomial':
            y = torch.LongTensor(y)
        elif self.loss == 'bin_xent':
            y = torch.Tensor(y)

        # initialize inner model
        self.logreg_ = LitLogisticRegression(X.shape[1],
                                             y.shape[1],
                                             l2_alpha=self.alpha,
                                             loss=self.loss)
        # initialize pytorch dataset / dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers)

        gpus = None
        accelerator = 'cpu'
        if self.accelerator.startswith('cuda'):
            device_nr = int(self.accelerator.split(':')[-1])
            gpus = [device_nr]
            accelerator = 'gpu'

        trainer = pl.Trainer(accelerator=accelerator,
                             max_epochs=self.max_iter,
                             enable_checkpointing=False,
                             enable_model_summary=False,
                             enable_progress_bar=self.verbose,
                             auto_select_gpus=False,
                             devices=gpus)
        trainer.fit(self.logreg_, train_dataloaders=train_loader)

        return self

    def predict_proba(self, X):
        """
        """
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        """
        """
        # check is fit had been called
        check_is_fitted(self, 'logreg_')

        # input validation
        X = check_array(X)

        X = torch.Tensor(X)
        y_hat = self.logreg_._model(X)
        y_hat = F.logsigmoid(y_hat)

        return y_hat.detach().cpu().numpy()

    def predict(self, X):
        """
        """
        p = self.predict_proba(X)
        if self.loss == 'multinomial':
            y = np.argmax(p, axis=1)
        elif self.loss == 'bin_xent':
            y = p
        else:
            raise ValueError(
                '[ERROR] only `bin_xent` and `multinomial` is supported!'
            )
        return y
