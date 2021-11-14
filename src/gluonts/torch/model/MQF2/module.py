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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from gluonts.core.component import validated
from gluonts.torch.model.deepar.module import DeepARModel, LaggedLSTM

from gluonts.torch.modules.distribution_output import DistributionOutput
from gluonts.torch.distributions.MQF2Output import MQF2DistributionOutput
from .icnn_utils import ActNorm, SequentialFlow, DeepConvexFlow, PICNN, CPFlow, SequentialCPFlow

class MultivariateDeepARModel(DeepARModel):
    @validated()
    def __init__(
            self,
            freq: str,
            context_length: int,
            prediction_length: int,
            num_feat_dynamic_real: int,
            num_feat_static_real: int,
            num_feat_static_cat: int,
            cardinality: List[int],
            distr_output: DistributionOutput,
            embedding_dimension: Optional[List[int]] = None,
            num_layers: int = 2,
            hidden_size: int = 40,
            dropout_rate: float = 0.1,
            lags_seq: Optional[List[int]] = None,
            scaling: bool = True,
            num_parallel_samples: int = 100,
            icnn_hidden_size: int = 64,
            icnn_num_layers: int = 5,
            is_energy_score: bool = True,
            threshold_input=100,
            es_num_samples=50,
            estimate_logdet: bool = False,
    ) -> None:
        super().__init__(
        freq=freq,
        context_length=context_length,
        prediction_length=prediction_length,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_real=num_feat_static_real,
        num_feat_static_cat=num_feat_static_cat,
        cardinality=cardinality,
        embedding_dimension=embedding_dimension,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        distr_output=distr_output,
        lags_seq=lags_seq,
        scaling=scaling,
        num_parallel_samples=num_parallel_samples,
    )

        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples

        icnn = PICNN(dim=self.prediction_length, dimh=icnn_hidden_size, dimc=hidden_size, num_hidden_layers=icnn_num_layers, symm_act_first=True,
                     softplus_type='gaussian_softplus',
                     zero_softplus=True, is_energy_score=is_energy_score)

        # convexflow = DeepConvexFlow(icnn, self.prediction_length, unbiased=False, no_identity_term=is_energy_score)
        convexflow = CPFlow(icnn, self.prediction_length, unbiased=False, is_energy_score=is_energy_score, estimate_logdet=estimate_logdet)

        if is_energy_score:
            layers = [convexflow]
        else:
            layers = [ActNorm(self.prediction_length),
                      convexflow,
                      ActNorm(self.prediction_length),
            ]

        self.flow = SequentialCPFlow(layers)

    def unroll_lagged_rnn(
            self,
            feat_static_cat: torch.Tensor,
            feat_static_real: torch.Tensor,
            past_time_feat: torch.Tensor,
            past_target: torch.Tensor,
            past_observed_values: torch.Tensor,
            future_time_feat: Optional[torch.Tensor] = None,
            future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[SequentialFlow, torch.Tensor],
        torch.Tensor,
    ]:

        _, scale, output, _, _ = super().unroll_lagged_rnn(feat_static_cat=feat_static_cat,
        feat_static_real=feat_static_real,
        past_time_feat=past_time_feat,
        past_target=past_target,
        past_observed_values=past_observed_values,
        future_time_feat=future_time_feat,
        future_target=future_target)

        params = (self.flow, output[:, :self.context_length])

        return params, scale

    @torch.jit.ignore
    def output_distribution(
            self, params, scale=None, inference=False,
    ) -> torch.distributions.Distribution:
        sliced_params = list(params)
        if inference:
            sliced_params = [params[0]] + [p[:, -1] for p in params[1:]]

        # sliced_params = sliced_params + [self.prediction_length, self.threshold_input, self.es_num_samples]

        return self.distr_output.distribution(*sliced_params, scale=scale)

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            feat_static_real: torch.Tensor,
            past_time_feat: torch.Tensor,
            past_target: torch.Tensor,
            past_observed_values: torch.Tensor,
            future_time_feat: torch.Tensor,
            num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        distr = self.output_distribution(params, inference=True)

        unscaled_future_samples = distr.sample(sample_shape=(self.num_parallel_samples,))

        return unscaled_future_samples * scale.unsqueeze(-1)
