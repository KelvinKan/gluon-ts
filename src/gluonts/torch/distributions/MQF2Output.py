from typing import Dict, Optional, Tuple, List, cast

import torch
import torch.nn.functional as F
import numpy

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import (
    Distribution,
    DistributionOutput,
)

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class MQF2Distribution(Distribution):
    def __init__(
        self,
        flow,
        hidden_state,
        prediction_length,
        threshold_input,
        es_num_samples,
        is_energy_score: bool = True,
        beta: float = 1.0,
        validate_args=False,
    ) -> None:

        self.flow = flow
        self.hidden_state = hidden_state
        self.prediction_length = prediction_length
        self.is_energy_score = is_energy_score

        # self.num_windows = (
        #     self.hidden_state.shape[1] if len(self.hidden_state.shape) == 3 else 1
        # )


        #ToDo: move it to module
        self.num_sliding_windows = self.hidden_state.shape[-2] if len(self.hidden_state.shape) > 2 else 1
        # self.numel_batch = torch.numel(self.hidden_state[..., 0])
        self.numel_batch = self.get_numel(self.batch_shape) #torch.prod(torch.tensor(self.batch_shape)).item()

        # self.batch_size = self.hidden_state.shape[0] #self.batch_shape[:-1]

        # for seq_flow in self.flow.flows:
        #     try:
        #         self.prediction_length = seq_flow.icnn.Wzs[0].in_features
        #         break
        #     except:
        #         pass


        # identity_matrix = torch.eye(self.prediction_length,
        #                          dtype=self.hidden_state.dtype, device=self.hidden_state.device,
        #                          layout=self.hidden_state.layout)

        # need to check if this works for gpu
        identity_matrix = torch.eye(self.prediction_length, out=torch.empty_like(self.hidden_state))
        zero_vector = torch.zeros_like(identity_matrix[0])

        self.standard_Gaussian = MultivariateNormal(zero_vector, identity_matrix)
        zero = torch.tensor(0,
            dtype=hidden_state.dtype,
            device=hidden_state.device)
        one = torch.ones_like(zero)
        self.standard_normal = Normal(zero, one)

        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples
        self.beta = beta

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_energy_score:
            return self.energy_score(z)
        else:
            return -self.log_prob(z, context=self.hidden_state)

    def reshape_input(self, z: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction_length = self.prediction_length
        z = z.unfold(dimension=-1, size=prediction_length, step=1)
        z = z.reshape(-1, z.shape[-1])
        hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])

        return z, hidden_state

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:

        hidden_state = self.hidden_state
        flow = self.flow

        z = torch.clamp(z, min=-self.threshold_input, max=self.threshold_input)
        # z = torch.min(z, self.threshold_input * torch.ones_like(z))

        # z = z.unfold(dimension=-1, size=prediction_length, step=1)
        # z = z.reshape(-1, z.shape[-1])
        #
        # hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])

        z, hidden_state = self.reshape_input(z, hidden_state)

        loss = flow.logp(z, context=hidden_state)

        return loss

    # def energy_score(self, z: torch.Tensor) -> torch.Tensor:
    #     # es_num_samples is the number of samples drawn for each of X, X', X'' in energy score approximation
    #     es_num_samples, numel_batch = self.es_num_samples, self.numel_batch
    #     prediction_length, hidden_state = self.prediction_length, self.hidden_state
    #     standard_Gaussian = self.standard_Gaussian
    #     flow = self.flow
    #     beta = self.beta
    #
    #     # Tomorrow: move to Sequential flow
    #     # cond_num = self.hidden_state.shape[0]*self.hidden_state.shape[1]
    #     # cond_num = torch.prod(torch.tensor(batch_shape)).item()
    #     total_sample_num = es_num_samples*numel_batch
    #     # z = z.unfold(dimension=-1, size=prediction_length, step=1)
    #     # z = z.reshape(-1, z.shape[-1]).repeat_interleave(repeats=es_num_samples, dim=0)
    #     # hidden_state_repeat = hidden_state.reshape(-1, hidden_state.shape[-1]).repeat_interleave(repeats=es_num_samples, dim=0)
    #
    #     z, hidden_state = self.reshape_input(z, hidden_state)
    #
    #     z_repeat = z.repeat_interleave(repeats=es_num_samples, dim=0)
    #     hidden_state_repeat = hidden_state.repeat_interleave(repeats=es_num_samples, dim=0)
    #
    #     X = flow(standard_Gaussian.sample([total_sample_num]), context=hidden_state_repeat)
    #     X_prime = flow(standard_Gaussian.sample([total_sample_num]), context=hidden_state_repeat)
    #     X_bar = flow(standard_Gaussian.sample([total_sample_num]), context=hidden_state_repeat)
    #
    #     # first_term = (
    #     #     torch.mean(
    #     #     (
    #     #             torch.norm(
    #     #     X.view(numel_batch, 1, es_num_samples, prediction_length)
    #     #     - X_prime.view(numel_batch, es_num_samples, 1, prediction_length), dim=-1
    #     # )**beta).view(numel_batch, -1), dim=-1)
    #     # )
    #
    #     first_term = torch.norm(
    #         X.view(numel_batch, 1, es_num_samples, prediction_length)
    #         - X_prime.view(numel_batch, es_num_samples, 1, prediction_length), dim=-1
    #     ) ** beta
    #
    #     mean_first_term = torch.mean(first_term.view(numel_batch, -1), dim=-1)
    #
    #     # second_term = torch.mean((torch.norm(
    #     #     X_bar.view(numel_batch, es_num_samples, prediction_length)
    #     #     - z_repeat.view(numel_batch, es_num_samples, prediction_length), dim=-1
    #     # )**beta).view(numel_batch, -1), dim=-1)
    #
    #     second_term = torch.norm(
    #         X_bar.view(numel_batch, es_num_samples, prediction_length)
    #         - z_repeat.view(numel_batch, es_num_samples, prediction_length), dim=-1
    #     ) ** beta
    #
    #     mean_second_term = torch.mean(second_term.view(numel_batch, -1), dim=-1)
    #
    #     # loss = -0.5*first_term + second_term
    #     # loss = -0.5*first_term + second_term
    #     loss = -0.5*mean_first_term + mean_second_term
    #
    #     return loss

    def energy_score(self, z: torch.Tensor) -> torch.Tensor:
        # es_num_samples is the number of samples drawn for each of X, X', X'' in energy score approximation
        es_num_samples, numel_batch = self.es_num_samples, self.numel_batch
        prediction_length, hidden_state = self.prediction_length, self.hidden_state
        standard_Gaussian = self.standard_Gaussian
        flow = self.flow
        beta = self.beta

        # Tomorrow: move to Sequential flow
        # cond_num = self.hidden_state.shape[0]*self.hidden_state.shape[1]
        # cond_num = torch.prod(torch.tensor(batch_shape)).item()
        total_sample_num = es_num_samples*numel_batch
        # z = z.unfold(dimension=-1, size=prediction_length, step=1)
        # z = z.reshape(-1, z.shape[-1]).repeat_interleave(repeats=es_num_samples, dim=0)
        # hidden_state_repeat = hidden_state.reshape(-1, hidden_state.shape[-1]).repeat_interleave(repeats=es_num_samples, dim=0)

        z, hidden_state = self.reshape_input(z, hidden_state)

        loss = flow.energy_score(z, hidden_state, es_num_samples=es_num_samples, beta=beta)

        return loss

    # def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
    #     numel_batch, batch_shape = self.numel_batch, self.batch_shape
    #     hidden_state = self.hidden_state
    #     flow = self.flow
    #     standard_Gaussian = self.standard_Gaussian
    #
    #     num_sample_per_batch = torch.prod(torch.tensor(sample_shape)).item()
    #     num_sample = num_sample_per_batch * numel_batch
    #
    #     hidden_state_repeat = hidden_state.repeat_interleave(repeats=num_sample_per_batch, dim=0)
    #
    #     # Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample])
    #     # Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample_per_batch])
    #     Gaussian_sample = standard_Gaussian.sample([num_sample])
    #
    #     if self.is_energy_score:
    #         sample = flow(Gaussian_sample, context=hidden_state_repeat)
    #     else:
    #         sample = flow.reverse(Gaussian_sample, context=hidden_state_repeat)
    #
    #     # sample = sample.reshape((self.batch_size,) + sample_shape + (-1,))
    #     sample = sample.reshape(batch_shape + sample_shape + (-1,))
    #
    #     return sample

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        numel_batch = self.numel_batch
        hidden_state = self.hidden_state
        prediction_length = self.prediction_length

        num_sample_per_batch = self.get_numel(sample_shape) #torch.prod(torch.tensor(sample_shape)).item()
        num_sample = num_sample_per_batch * numel_batch

        hidden_state_repeat = hidden_state.repeat_interleave(repeats=num_sample_per_batch, dim=0)

        # Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample])
        # Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample_per_batch])
        # Gaussian_sample = standard_Gaussian.sample([num_sample])

        alpha = torch.rand(
            (num_sample, prediction_length),
            dtype=self.hidden_state.dtype,
            device=self.hidden_state.device,
            layout=self.hidden_state.layout,
        )

        return self.quantile(alpha, hidden_state_repeat)

        # if self.is_energy_score:
        #     sample = flow(Gaussian_sample, context=hidden_state_repeat)
        # else:
        #     sample = flow.reverse(Gaussian_sample, context=hidden_state_repeat)
        #
        # # sample = sample.reshape((self.batch_size,) + sample_shape + (-1,))
        # sample = sample.reshape(batch_shape + sample_shape + (-1,))
        #
        # return sample

    def quantile(self, alpha: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        flow = self.flow
        if hidden_state is None:
            hidden_state = self.hidden_state

        #alpha shape = (batch_size, prediction_length)
        normal_quantile = self.standard_normal.icdf(alpha)

        if self.is_energy_score:
            result = flow(normal_quantile, context=hidden_state)
        else:
            result = flow.reverse(normal_quantile, context=hidden_state)

        return result

    def get_numel(self, tensor_shape: torch.Size()):
        return torch.prod(torch.tensor(tensor_shape)).item()

    @property
    def batch_shape(self) -> torch.Size():
        # last 2 dimensions are length of time series and hidden state size
        return self.hidden_state.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0


class MQF2DistributionOutput(DistributionOutput):
    distr_cls: type = MQF2Distribution

    @validated()
    def __init__(self, prediction_length: int, is_energy_score: bool = True, threshold_input: int = 100, es_num_samples:int = 50, beta: float = 1.0) -> None:
        super().__init__(self)
        # A null args_dim to be called by PtArgProj
        self.args_dim = cast(
            Dict[str, int],
            {"null": 1},
        )

        self.prediction_length = prediction_length
        self.is_energy_score = is_energy_score
        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples
        self.beta = beta

    @classmethod
    def domain_map(
        cls,
        hidden_state,
    ):
        # A null function to be called by ArgProj
        return ()

    def distribution(
        self,
        flow,
        hidden_state,
        # prediction_length,
        # threshold_input,
        # es_num_samples,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> MQF2Distribution:
        if scale is None:
            return self.distr_cls(flow, hidden_state, prediction_length=self.prediction_length, threshold_input=self.threshold_input, es_num_samples=self.es_num_samples, is_energy_score=self.is_energy_score,
                                  beta=self.beta,)
        else:
            distr = self.distr_cls(flow, hidden_state, prediction_length=self.prediction_length, threshold_input=self.threshold_input, es_num_samples=self.es_num_samples, is_energy_score=self.is_energy_score,
                                   beta=self.beta,)
            return TransformedMQF2Distribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedMQF2Distribution(TransformedDistribution):
    @validated()
    def __init__(self, base_distribution: MQF2Distribution,
            transforms: List[AffineTransform], validate_args=None,
    ) -> None:
        super().__init__(
            base_distribution, transforms, validate_args=validate_args
        )

    def scale_input(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale

        return z, scale

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        prediction_length = self.base_dist.prediction_length
        # z = y
        # scale = 1.0
        # for t in self.transforms[::-1]:
        #     assert isinstance(t, AffineTransform), "Not an AffineTransform"
        #     z = t._inverse(y)
        #     scale *= t.scale
        z, scale = self.scale_input(y)

        p = self.base_dist.log_prob(z)

        repeated_scale = scale.squeeze(-1).repeat_interleave(self.base_dist.num_sliding_windows, 0)

        # the log scale term can be omitted in optimization because it is a constant
        # prediction_length is the dimension of each sample
        return p - prediction_length * torch.log(repeated_scale)

    def energy_score(self, y:torch.Tensor) -> torch.Tensor:
        # z = y
        # scale = 1.0
        # for t in self.transforms[::-1]:
        #     assert isinstance(t, AffineTransform), "Not an AffineTransform"
        #     z = t._inverse(y)
        #     scale *= t.scale
        z, scale = self.scale_input(y)
        loss = self.base_dist.energy_score(z)
        repeated_scale = scale.squeeze(-1).repeat_interleave(self.base_dist.num_sliding_windows, 0)
        return loss * (repeated_scale ** self.base_dist.beta)

        #verify that repeated_scale is better than scale