# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from icnn_lib.flows import SequentialFlow, DeepConvexFlow, ActNorm
from icnn_lib.icnn import PICNN as ConvexNet
from icnn_lib.icnn import softplus
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

class DeepConvexNet(DeepConvexFlow):
    r"""
    Class that takes a partially input convex neural network (picnn) as input and equips it with functions of logdet
    computation (both estimation and exact computation)

    Parameters
    ----------
    picnn
        A SequentialNet instance of a partially input convex neural network (picnn)
    dim
        Dimension of the input
    is_energy_score
        Indicates if energy score is used as the objective function
        If yes, the network is not required to be strictly convex, so we can just use the picnn
        otherwise, a quadratic term is added to the output of picnn to render it strictly convex
    m1
        Dimension of the Krylov subspace of the Lanczos tridiagonalization used in approximating H of logdet(H)
    m2
        Iteration number of the conjugate gradient algorithm used to approximate logdet(H)
    rtol
        relative tolerance of the conjugate gradient algorithm
    atol
        absolute tolerance of the conjugate gradient algorithm

    """

    def __init__(self, picnn, dim, is_energy_score=False, estimate_logdet=False, m1=10, m2=None, rtol=0.0, atol=1e-3):
        super().__init__(picnn,
                         dim,
                         m1=m1,
                         m2=m2,
                         rtol=rtol,
                         atol=atol,)

        self.picnn = self.icnn
        self.is_energy_score = is_energy_score
        self.estimate_logdet = estimate_logdet

    def get_potential(self, x, context=None):
        n = x.size(0)
        picnn = self.picnn(x, context)

        if self.is_energy_score:
            return picnn
        else:
            return F.softplus(self.w1) * picnn + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        if self.estimate_logdet:
            return self.forward_transform_stochastic(x, logdet, context=context, extra=extra)
        else:
            return self.forward_transform_bruteforce(x, logdet, context=context)

class SequentialNet(SequentialFlow):
    r"""
    Class that combines a list of DeepConvexNet and ActNorm

    Parameters
    ----------
    networks
        list of DeepConvexNet and/or ActNorm instances

    """

    def __init__(self, networks):
        super().__init__(networks)
        self.networks = self.flows

    def forward(self, x, context=None):
        for network in self.networks:
            if isinstance(network, DeepConvexNet):
                x = network.forward(x, context=context)
            else:
                x = network.forward(x)
        return x

    def es_sample(self, hidden_state: torch.Tensor, dimension: int):
        """
        Auxiliary function for energy score computation

        It draws samples conditioned on the hidden state

        Parameters
        ----------
        hidden_state
            hidden_state which the samples conditioned on (num_samples, hidden_size)
        dimension
            dimension of the input

        Returns
        -------
        samples
            samples drawn (num_samples, dimension)
        """

        num_samples = hidden_state.shape[0]

        zero = torch.tensor(0,
                            dtype=hidden_state.dtype,
                            device=hidden_state.device)
        one = torch.ones_like(zero)
        standard_normal = Normal(zero, one)

        samples = self.forward(
            standard_normal.sample([num_samples*dimension]).view(num_samples,dimension),
            context=hidden_state
        )

        return samples

    def energy_score(self, z: torch.Tensor, hidden_state: torch.Tensor, es_num_samples: int = 50, beta: float = 1.0):
        """
        Computes the (approximated) energy score \sum_i ES(g,z_i),
        where ES(g,z_i) =
        -1/(2*es_num_samples^2) * \sum_{X,X'} ||X-X'||_2^beta + 1/es_num_samples *\sum_{X''} ||X''-z_i||_2^beta,
        X's are samples drawn from the quantile function g(\cdot, h_i) (gradient of picnn),
        h_i is the hidden state associated with z_i,
        and es_num_samples is the number of samples drawn for each of X, X', X'' in energy score approximation

        Parameters
        ----------
        z
            Observations (numel_batch, dimension)
        hidden_state
            Hidden state (numel_batch, hidden_size)
        es_num_samples
            Number of samples drawn for each of X, X', X'' in energy score approximation
        beta
            Hyperparameter of the energy score, see the formula above
        Returns
        -------
        loss
            energy score (numel_batch)
        """

        numel_batch, dimension = z.shape[0], z.shape[1]
        total_num_samples = es_num_samples * numel_batch

        z_repeat = z.repeat_interleave(repeats=es_num_samples, dim=0)
        hidden_state_repeat = hidden_state.repeat_interleave(repeats=es_num_samples, dim=0)

        X = self.es_sample(hidden_state_repeat, dimension)
        X_prime = self.es_sample(hidden_state_repeat, dimension)
        X_bar = self.es_sample(hidden_state_repeat, dimension)

        first_term = torch.norm(
            X.view(numel_batch, 1, es_num_samples, dimension)
            - X_prime.view(numel_batch, es_num_samples, 1, dimension), dim=-1
        ) ** beta

        mean_first_term = torch.mean(first_term.view(numel_batch, -1), dim=-1)

        second_term = torch.norm(
            X_bar.view(numel_batch, es_num_samples, dimension)
            - z_repeat.view(numel_batch, es_num_samples, dimension), dim=-1
        ) ** beta

        mean_second_term = torch.mean(second_term.view(numel_batch, -1), dim=-1)

        loss = -0.5 * mean_first_term + mean_second_term

        return loss