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

from torch.distributions.multivariate_normal import MultivariateNormal

class ICNNESDistribution(Distribution):
    def __init__(
        self,
        flow,
        hidden_state,
        threshold_input,
        num_expected,
        validate_args=False,
    ) -> None:
        self.flow = flow
        self.hidden_state = hidden_state

        self.batch_size = self.hidden_state.shape[0]

        # self.prediction_length = self.flow.flows[1].icnn.Wzs[0].in_features

        for seq_flow in self.flow.flows:
            try:
                self.prediction_length = seq_flow.icnn.Wzs[0].in_features
                break
            except:
                pass

        mean_tensor = torch.zeros(self.prediction_length,
                                  dtype=self.hidden_state.dtype, device=self.hidden_state.device,
                                  layout=self.hidden_state.layout)

        covar_tensor = torch.eye(self.prediction_length,
                                 dtype=self.hidden_state.dtype, device=self.hidden_state.device,
                                 layout=self.hidden_state.layout)


        self.standard_Gaussian = MultivariateNormal(mean_tensor, covar_tensor)

        self.threshold_input = threshold_input
        self.num_expected = num_expected

        # self.batch_shape = self.gamma.shape
        super(ICNNESDistribution, self).__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

    # To-do: quantile, rsample

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy_score(z)

    def energy_score(self, z: torch.Tensor) -> torch.Tensor:
        # num_expected is the number of samples drawn for each of X, X', X'' in energy score approximation
        num_expected = self.num_expected

        if len(self.hidden_state.shape) == 2:
            cond_num = self.hidden_state.shape[0]
            total_sample_num = num_expected*cond_num
        else:
            cond_num = self.hidden_state.shape[0]*self.hidden_state.shape[1]
            total_sample_num = num_expected*cond_num
            z = z.unfold(1, self.prediction_length, 1)
            z = z.reshape(-1, z.shape[-1]).repeat_interleave(repeats=num_expected, dim=0)
            hidden_state = self.hidden_state.reshape(-1, self.hidden_state.shape[-1]).repeat_interleave(repeats=num_expected, dim=0)

        X = self.flow(self.standard_Gaussian.sample([total_sample_num]), context=hidden_state)
        X_prime = self.flow(self.standard_Gaussian.sample([total_sample_num]), context=hidden_state)
        X_bar = self.flow(self.standard_Gaussian.sample([total_sample_num]), context=hidden_state)

        first_term = -0.5*torch.mean(torch.norm(
            X.view(cond_num, 1, num_expected, self.prediction_length)
            - X_prime.view(cond_num, num_expected, 1, self.prediction_length), dim=-1
        ).view(cond_num, -1), dim=-1)

        second_term = torch.mean(torch.norm(
            X_bar.view(cond_num, num_expected, self.prediction_length)
            - z.view(cond_num, num_expected, self.prediction_length), dim=-1
        ).view(cond_num, -1), dim=-1)

        loss = first_term + second_term

        return loss


    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:

        num_sample = torch.prod(torch.tensor(sample_shape)).item()

        hidden_state = self.hidden_state.repeat_interleave(repeats=num_sample, dim=0)

        Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample])

        # sample = self.flow.reverse(Gaussian_sample, context=hidden_state)
        sample = self.flow(Gaussian_sample, context=hidden_state)
        sample = sample.reshape((self.batch_size,) + sample_shape + (-1,))

        return sample

    @property
    def batch_shape(self) -> torch.Size():
        return self.hidden_state.shape[:-1]


class ICNNESDistributionOutput(DistributionOutput):
    distr_cls: type = ICNNESDistribution

    @validated()
    def __init__(self) -> None:
        super().__init__(self)
        self.args_dim = cast(
            Dict[str, int],
            {"dummy": 1},
        )

    @classmethod
    def domain_map(
        cls,
        flow,
        hidden_state,
        threshold_input,
        num_expected,
    ):

        return flow, hidden_state, threshold_input, num_expected

    def distribution(
        self,
        flow,
        hidden_state,
        threshold_input,
        num_expected,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> ICNNESDistribution:
        if scale is None:
            return self.distr_cls(flow, hidden_state, threshold_input, num_expected)
        else:
            distr = self.distr_cls(flow, hidden_state, threshold_input, num_expected)
            return TransformedICNNESDistribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedICNNESDistribution(TransformedDistribution):
    @validated()
    def __init__(self, base_distribution: ICNNESDistribution,
            transforms: List[AffineTransform], validate_args=None,
    ) -> None:
        # super().__init__(
        #     base_distribution, transforms, validate_args=validate_args
        # )
        self.base_dist = base_distribution
        self.transforms = transforms

    def energy_score(self, y:torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale
        loss = self.base_dist.energy_score(z)
        return loss * scale