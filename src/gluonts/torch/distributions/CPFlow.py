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

class CPFlowDistribution(Distribution):
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

        self.num_windows = (
            self.hidden_state.shape[1] if len(self.hidden_state.shape) == 3 else 1
        )

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
        super(CPFlowDistribution, self).__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

    # To-do: quantile, rsample

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(z, context=self.hidden_state)

    # In NegativeLogLikelihood, a negative sign will be assigned to log_prob
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # for idx, param in enumerate(self.flow.flows[1].icnn.Wczs[0].parameters()):
        #     print("getting parameters here")
        #     print(idx, param)

        if len(self.hidden_state.shape) == 2:
            return self.flow.logp(z, context=self.hidden_state)
        else:
            # num_windows = self.hidden_state.shape[1]
            # prediction_length = z.shape[1] - num_windows + 1

            z = z.unfold(1, self.prediction_length, 1)
            z = z.reshape(-1, z.shape[-1])

            hidden_state = self.hidden_state.reshape(-1, self.hidden_state.shape[-1])

            loss = self.flow.logp(z, context=hidden_state)

            return loss #.view(self.hidden_state.shape[0], self.hidden_state.shape[1], prediction_length)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:

        num_sample = torch.prod(torch.tensor(sample_shape)).item()

        hidden_state = self.hidden_state.repeat_interleave(repeats=num_sample, dim=0)

        Gaussian_sample = self.standard_Gaussian.sample([self.batch_size * num_sample])

        sample = self.flow.reverse(Gaussian_sample, context=hidden_state)
        sample = sample.reshape((self.batch_size,) + sample_shape + (-1,))

        return sample

    @property
    def batch_shape(self) -> torch.Size():
        return self.hidden_state.shape[:-1]


class CPFlowDistributionOutput(DistributionOutput):
    distr_cls: type = CPFlowDistribution

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
    ) -> CPFlowDistribution:
        if scale is None:
            return self.distr_cls(flow, hidden_state, threshold_input, num_expected)
        else:
            distr = self.distr_cls(flow, hidden_state, threshold_input, num_expected)
            return TransformedCPFlowDistribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedCPFlowDistribution(TransformedDistribution):
    @validated()
    def __init__(self, base_distribution: CPFlowDistribution,
            transforms: List[AffineTransform], validate_args=None,
    ) -> None:
        # super().__init__(
        #     base_distribution, transforms, validate_args=validate_args
        # )
        self.base_dist = base_distribution
        self.transforms = transforms

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        dimension = self.base_dist.prediction_length
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale

        z = torch.min(z, self.base_dist.threshold_input*torch.ones_like(z))

        p = self.base_dist.log_prob(z)

        # print(f"max_sample: {torch.max(y)}, min_sample: {torch.min(y)}")

        repeated_scale = scale.squeeze(-1).repeat_interleave(self.base_dist.num_windows, 0)

        return p - dimension * torch.log(repeated_scale)

        # the log scale term can be omitted in optimization because it is a constant