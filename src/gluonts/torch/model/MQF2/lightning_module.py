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

import torch

from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.model.deepar.lightning_module import DeepARLightningModule
from gluonts.torch.model.MQF2.module import MultivariateDeepARModel

class MultivariateDeepARLightningModule(DeepARLightningModule):
    def __init__(
        self,
        model: MultivariateDeepARModel,
        # prediction_length: int,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
        )
        # self.prediction_length = prediction_length

    def _compute_loss(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        # prediction_length = self.prediction_length

        params, scale = self.model.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        distr = self.model.output_distribution(params, scale)

        context_target = past_target[:, -self.model.context_length + 1:]
        target = torch.cat(
            (context_target, future_target),
            dim=1,
        )

        # prediction_length = distr.prediction_length

        # target = target.unfold(dimension=-1, size=prediction_length, step=1)
        # target = target.reshape(-1, target.shape[-1])
        # z = z.unfold(dimension=-1, size=prediction_length, step=1)
        # z = z.reshape(-1, z.shape[-1])

        loss_values = self.loss(distr, target)

        return loss_values.mean()