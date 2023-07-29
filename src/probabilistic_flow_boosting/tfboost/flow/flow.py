from typing import List, Union

import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nflows.distributions import ConditionalDiagonalNormal
from torch.utils.data import DataLoader, TensorDataset

from .odefunc import ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


class ContinuousNormalizingFlow:

    def __init__(
            self,
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False
    ):
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.flow = self.build_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            context_dim=context_dim,
            num_blocks=num_blocks,
            conditional=conditional,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            time_length=time_length,
            train_T=train_T,
            solver=solver,
            atol=atol,
            rtol=rtol,
            use_adjoint=use_adjoint,
            batch_norm=batch_norm,
            bn_lag=bn_lag,
            sync_bn=sync_bn
        )
        self.flow = self.flow.to(self.DEVICE)
        self.context_encoder = nn.Identity().to(self.DEVICE)
        self.distribution = ConditionalDiagonalNormal(shape=[input_dim], context_encoder=nn.Identity()).to(self.DEVICE)
        self.losses = {"train": [], "val": []}
        self.epoch_best = -1

    @staticmethod
    def build_model(
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False
    ):
        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
            )
            cnf = CNF(
                odefunc=odefunc,
                T=time_length,
                train_T=train_T,
                conditional=conditional,
                solver=solver,
                use_adjoint=use_adjoint,
                atol=atol,
                rtol=rtol,
            )
            return cnf

        chain = [build_cnf() for _ in range(num_blocks)]
        if batch_norm:
            bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn) for _ in range(num_blocks)]
            bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn)]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain
        model = SequentialFlow(chain)
        return model

    def setup_context_encoder(self, context_encoder: nn.Module):
        self.context_encoder = context_encoder.to(self.DEVICE)

    def fit(self, X: np.ndarray, context: np.ndarray, params: np.ndarray, X_val: Union[np.ndarray, None] = None,
            context_val: Union[np.ndarray, None] = None, params_val: Union[np.ndarray, None] = None,
            n_epochs: int = 100, batch_size: int = 1000, verbose: bool = False):
        X_t: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.DEVICE)
        context_t: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params_t: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

        dataset: DataLoader = DataLoader(
            dataset=TensorDataset(X_t, context_t, params_t),
            shuffle=True,
            batch_size=batch_size
        )

        self.optimizer = optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()))

        mid = uuid.uuid4()  # To be able to run multiple experiments in parallel.
        loss_best = np.inf

        for i in range(n_epochs):
            for x, c, p in dataset:
                self.optimizer.zero_grad()

                logpx = self._log_prob(x, c, p)
                loss = -logpx.mean()

                loss.backward()
                self.optimizer.step()

            self._log(X, context, params, mode="train", batch_size=batch_size, verbose=verbose)

            if X_val is not None and context_val is not None and params_val is not None:
                loss_val = self._log(X_val, context_val, params_val, mode="val", batch_size=batch_size,
                                     verbose=verbose)
                # Save model if better
                if loss_val < loss_best:
                    self.epoch_best = i
                    loss_best = loss_val
                    self._save_temp(i, mid)

        if X_val is not None and context_val is not None and params_val is not None:
            return self._load_temp(self.epoch_best, mid)
        return self

    def _log_prob(self, X: torch.Tensor, context: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probability on data batch.
        :param X: 
        :param context: 
        :param params: 
        :return: 
        """
        zero: torch.Tensor = torch.zeros(X.shape[0], 1, device=X.device)
        context_e: torch.Tensor = self.context_encoder(context)
        z, delta_logp = self.flow(x=X, context=context_e, logpx=zero)

        logpz: torch.Tensor = self.distribution.log_prob(z, params)
        logpz: torch.Tensor = logpz.reshape(-1, 1)
        logpx: torch.Tensor = logpz - delta_logp
        return logpx

    @torch.no_grad()
    def log_prob(self, X: np.ndarray, context: np.ndarray, params: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        Calculate log probability of sample given data, context and prior distribution parameters.
        :param X: Data.
        :param context: Data context (Conditioning to flow).
        :param params: Prior distribution parameters.
        :param batch_size: Batch size.
        :return: Log probability of sample given data, context and prior distribution parameters.
        """
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.DEVICE)
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

        dataset: DataLoader = DataLoader(
            dataset=TensorDataset(X, context, params),
            shuffle=False,
            batch_size=batch_size
        )

        logpxs: List[torch.Tensor] = [self._log_prob(X=x, context=c, params=p) for x, c, p in dataset]
        logpx: torch.Tensor = torch.cat(logpxs, dim=0)
        logpx: np.ndarray = logpx.detach().cpu().numpy()
        return logpx

    @torch.no_grad()
    def sample(self, context: np.ndarray, params: np.ndarray, num_samples: int = 10,
               batch_size: int = 1000) -> np.ndarray:
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

        dataset: DataLoader = DataLoader(
            dataset=TensorDataset(context, params),
            shuffle=False,
            batch_size=batch_size
        )

        all_samples: List[torch.Tensor] = []

        for c, p in dataset:
            samples: torch.Tensor = self.distribution.sample(num_samples=num_samples, context=p)
            context_e: torch.Tensor = self.context_encoder(c)
            samples: torch.Tensor = self.flow(x=samples, context=context_e, reverse=True)

            all_samples.append(samples)

        samples: torch.Tensor = torch.cat(all_samples, dim=0)
        samples: np.ndarray = samples.detach().cpu().numpy()
        return samples

    @torch.no_grad()
    def embed(self, context: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)

        dataset: DataLoader = DataLoader(
            dataset=TensorDataset(context),
            shuffle=False,
            batch_size=batch_size
        )

        context_e: List[torch.Tensor] = [self.context_encoder(c[0]) for c in dataset]
        context_e: torch.Tensor = torch.cat(context_e, dim=0)
        context_e: np.ndarray = context_e.detach().cpu().numpy()
        return context_e

    def _log(self, X: np.ndarray, context: np.ndarray, params: np.ndarray, mode: str, batch_size: int = 1000,
             verbose: bool = False):
        loss = -self.log_prob(X, context, params, batch_size=batch_size).mean()
        if verbose:
            print(f"{mode} loss: {loss}")
        self.losses[mode].append(loss)
        return loss

    def _save_temp(self, epoch, mid):
        print(f"Saving model from epoch {epoch}.")
        torch.save(self, f"/tmp/model_{mid}.pt")

    def _load_temp(self, epoch, mid):
        print(f"Loading model from epoch {epoch}.")
        return torch.load(f"/tmp/model_{mid}.pt")
