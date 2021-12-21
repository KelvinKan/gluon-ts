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

from typing import Dict, Optional, Tuple, List, cast

import torch
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import (
    Distribution,
    DistributionOutput,
)

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)


class BidirPiecewiseLinear(Distribution):
    def __init__(
        self,
        gamma: torch.Tensor,
        slopes_left: torch.Tensor,
        slopes_right: torch.Tensor,
        knots_left: torch.Tensor,
        knots_right: torch.Tensor,
        validate_args=False,
    ) -> None:
        self.gamma = gamma
        self.slopes_left, self.slopes_right = slopes_left, slopes_right
        self.knots_left, self.knots_right = knots_left, knots_right

        paras = BidirPiecewiseLinear._to_orig_params(
            slopes_left, slopes_right, knots_left, knots_right
        )
        (
            self.m_left,
            self.m_right,
            self.knots_pos_left,
            self.knots_pos_right,
        ) = paras

        batch_shape = self.gamma.shape
        super(BidirPiecewiseLinear, self).__init__(
            batch_shape, validate_args=validate_args
        )

    @staticmethod
    def _to_orig_params(
        slopes_left: torch.Tensor,
        slopes_right: torch.Tensor,
        knots_left: torch.Tensor,
        knots_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        m_left = BidirPiecewiseLinear.parametrize_slopes(slopes_left)
        m_right = BidirPiecewiseLinear.parametrize_slopes(slopes_right)

        knots_pos_left = BidirPiecewiseLinear.parametrize_knots(
            knots_left, left=True
        )
        knots_pos_right = BidirPiecewiseLinear.parametrize_knots(
            knots_right, left=False
        )

        return m_left, m_right, knots_pos_left, knots_pos_right

    @staticmethod
    def parametrize_slopes(slopes: torch.Tensor) -> torch.Tensor:
        slopes_parametrized = slopes
        slopes_parametrized[..., 1:] = torch.diff(slopes, dim=-1)
        return slopes_parametrized

    @staticmethod
    def parametrize_knots(
        knots: torch.Tensor, left: Optional[bool] = True
    ) -> torch.Tensor:
        if left:
            knots_pos = (
                0.5 - torch.cumsum(knots, dim=-1)[..., :-1] / 2
            )  # the last entry is 0
        else:
            knots_pos = (
                0.5 + torch.cumsum(knots, dim=-1)[..., :-1] / 2
            )  # the last entry is 1
        return knots_pos

    def quantile(self, u: torch.Tensor) -> torch.Tensor:
        return self.quantile_internal(u, dim=0)

    def quantile_internal(
        self, u: torch.Tensor, dim: Optional[int] = None
    ) -> torch.Tensor:
        # output shape = u.shape
        if dim is not None:
            # when num_samples!=None
            #
            # In testing
            # dim = 0
            # u.shape     = (num_samples, *batch_shape)
            # gamma.shape = (1, *batch_shape)
            # other_para.shape = (1, *batch_shape, num_knots_left or num_knots_right), because we split left and right
            #
            # In training, u_tilde is needed to compute CRPS
            # dim = -2
            # u.shape     = (*batch_shape, num_knots)
            # gamma.shape = (*batch_shape, 1)
            # other_para.shape = (*batch_shape, 1, num_knots_left or num_knots_right), because we split left and right

            gamma = self.gamma.unsqueeze(0 if dim == 0 else -1)  # gamma
            m_left, m_right = (
                self.m_left.unsqueeze(dim),
                self.m_right.unsqueeze(dim),
            )
            knots_pos_left, knots_pos_right = (
                self.knots_pos_left.unsqueeze(dim),
                self.knots_pos_right.unsqueeze(dim),
            )
        else:
            # when num_samples is None
            #
            # dim = None
            # u.shape          = (*batch)
            # gamma.shape      = (*batch)
            # other_para.shape = (*batch, num_knots_left or num_knots_right), because we split left and eight

            gamma = self.gamma
            m_left, m_right = self.m_left, self.m_right
            knots_pos_left, knots_pos_right = (
                self.knots_pos_left,
                self.knots_pos_right,
            )

        u = u.unsqueeze(-1)

        u_spline_left = F.relu(
            knots_pos_left - u
        )  # first knot appears at 0.5 (median)
        u_spline_right = F.relu(
            u - knots_pos_right
        )  # first knot appears at 0.5 (median)

        quantile = (
            gamma
            - torch.sum(m_left * u_spline_left, dim=-1)
            + torch.sum(m_right * u_spline_right, dim=-1)
        )

        return quantile

    def get_u_tilde(self, z: torch.Tensor) -> torch.Tensor:
        """
        compute the quantile levels u_tilde s.t. quantile(u_tilde)=z

        Input
        z: observations, shape = gamma.shape = (*batch_size)

        Output
        u_tilde: of type torch.Tensor
        """
        gamma = self.gamma
        m_left, m_right = self.m_left, self.m_right
        knots_pos_left, knots_pos_right = (
            self.knots_pos_left,
            self.knots_pos_right,
        )

        knots_eval_left = self.quantile_internal(knots_pos_left, dim=-2)
        knots_eval_right = self.quantile_internal(knots_pos_right, dim=-2)

        mask_left = torch.gt(knots_eval_left, z.unsqueeze(-1))
        mask_right = torch.lt(knots_eval_right, z.unsqueeze(-1))

        sum_slopes_left = torch.sum(mask_left * m_left, dim=-1)
        sum_slopes_right = torch.sum(mask_right * m_right, dim=-1)

        zero_val = torch.zeros(
            1, dtype=gamma.dtype, device=gamma.device, layout=gamma.layout
        )
        one_val = torch.ones(
            1, dtype=gamma.dtype, device=gamma.device, layout=gamma.layout
        )

        sum_slopes_left_nz = torch.where(
            sum_slopes_left == zero_val, one_val, sum_slopes_left
        )

        sum_slopes_right_nz = torch.where(
            sum_slopes_right == zero_val, one_val, sum_slopes_right
        )

        u_tilde = torch.where(
            sum_slopes_right == zero_val,
            (
                z
                - gamma
                + torch.sum(m_left * knots_pos_left * mask_left, dim=-1)
            )
            / sum_slopes_left_nz,
            (
                z
                - gamma
                + torch.sum(m_right * knots_pos_right * mask_right, dim=-1)
            )
            / sum_slopes_right_nz,
        )

        # If the value of z is median, then u_tilde is 0.5
        u_tilde = torch.where(
            torch.abs(z - gamma) < 1e-10, one_val / 2, u_tilde
        )

        u_tilde = torch.max(torch.min(u_tilde, one_val), zero_val)

        return u_tilde

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        target_shape = (
            self.gamma.shape
            if sample_shape == torch.Size()
            else torch.Size(sample_shape) + self.gamma.shape
        )

        u = torch.rand(
            target_shape,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
            layout=self.gamma.layout,
        )

        sample = self.quantile(u)

        if sample_shape == torch.Size():
            sample = sample.squeeze(dim=0)

        return sample

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.crps(z)

    def crps(self, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        m_left, m_right = self.m_left, self.m_right
        knots_pos_left, knots_pos_right = (
            self.knots_pos_left,
            self.knots_pos_right,
        )

        u_tilde = self.get_u_tilde(z)

        max_u_tilde_knots_left = torch.max(
            u_tilde.unsqueeze(-1), knots_pos_left
        )
        max_u_tilde_knots_right = torch.max(
            u_tilde.unsqueeze(-1), knots_pos_right
        )

        coeff_left = (
            2 * knots_pos_left * u_tilde.unsqueeze(-1)
            - u_tilde.unsqueeze(-1) ** 2
            + (knots_pos_left ** 3) / 3
            - 2 * knots_pos_left * max_u_tilde_knots_left
            + max_u_tilde_knots_left ** 2
        )

        coeff_right = (
            1 / 3
            - knots_pos_right
            - (knots_pos_right ** 3) / 3
            + 2 * knots_pos_right * max_u_tilde_knots_right
            - max_u_tilde_knots_right ** 2
        )

        result = (
            (2 * u_tilde - 1) * (z - gamma)
            + torch.sum(m_left * coeff_left, dim=-1)
            + torch.sum(m_right * coeff_right, dim=-1)
        )

        return result


class BidirPiecewiseLinearOutput(DistributionOutput):
    distr_cls: type = BidirPiecewiseLinear

    @validated()
    def __init__(self, num_pieces_left: int, num_pieces_right: int) -> None:
        super().__init__(self)

        assert (
            isinstance(num_pieces_left, int)
            and isinstance(num_pieces_right, int)
            and num_pieces_left > 1
            and num_pieces_left > 1
        ), "num_pieces should be an integer larger than 1"

        self.num_pieces_left, self.num_pieces_right = (
            num_pieces_left,
            num_pieces_right,
        )
        self.args_dim = cast(
            Dict[str, int],
            {
                "gamma": 1,
                "slopes_left": num_pieces_left,
                "slopes_right": num_pieces_right,
                "knots_left": num_pieces_left,
                "knots_right": num_pieces_right,
            },
        )

    @classmethod
    def domain_map(
        cls,
        gamma: torch.Tensor,
        slopes_left: torch.Tensor,
        slopes_right: torch.Tensor,
        knots_left: torch.Tensor,
        knots_right: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        slopes_left, slopes_right = torch.abs(slopes_left), torch.abs(
            slopes_right
        )
        knots_left, knots_right = F.softmax(knots_left, dim=-1), F.softmax(
            knots_right, dim=-1
        )

        # Need to pad 0.5s here because the knot pos start at 0.5
        zero_point_five = torch.zeros_like(knots_left[..., 0:1])
        knots_left = torch.cat([zero_point_five, knots_left], dim=-1)
        knots_right = torch.cat([zero_point_five, knots_right], dim=-1)

        return (
            gamma.squeeze(dim=-1),
            slopes_left,
            slopes_right,
            knots_left,
            knots_right,
        )

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> BidirPiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedBidirPiecewiseLinear(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedBidirPiecewiseLinear(TransformedDistribution):
    @validated()
    def __init__(
        self,
        base_distribution: BidirPiecewiseLinear,
        transforms: List[AffineTransform],
        validate_args=None,
    ) -> None:
        super().__init__(
            base_distribution, transforms, validate_args=validate_args
        )

    def crps(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale
        p = self.base_dist.crps(z)
        return p * scale
