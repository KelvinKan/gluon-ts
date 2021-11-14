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

from typing import List, Optional, Iterable, Dict, Any
from gluonts.torch.model.deepar.estimator import DeepAREstimator
from gluonts.torch.model.MQF2.lightning_module import MultivariateDeepARLightningModule
from gluonts.torch.model.MQF2.module import MultivariateDeepARModel

from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood, EnergyScore

from gluonts.torch.modules.distribution_output import DistributionOutput

from gluonts.torch.distributions.MQF2Output import MQF2DistributionOutput

from gluonts.time_feature import TimeFeature

class MultivariateDeepAREstimator(DeepAREstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        icnn_hidden_size: int = 64,
        icnn_num_layers: int = 5,
        is_energy_score=True,
        threshold_input = 100,
        es_num_samples = 50,
        beta: float = 1.0,
        estimate_logdet: bool = False,
    ) -> None:

        distr_output = MQF2DistributionOutput(prediction_length=prediction_length,
                                              is_energy_score=is_energy_score,
                                              threshold_input=threshold_input,
                                              es_num_samples=es_num_samples,
                                              beta=beta,
                                              )

        loss = EnergyScore() if is_energy_score else NegativeLogLikelihood()

        super().__init__(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_feat_static_cat=num_feat_static_cat,
        num_feat_static_real=num_feat_static_real,
        cardinality=cardinality,
        embedding_dimension=embedding_dimension,
        distr_output=distr_output,
        loss=loss,
        scaling=scaling,
        lags_seq=lags_seq,
        time_features=time_features,
        num_parallel_samples=num_parallel_samples,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer_kwargs=trainer_kwargs,
    )

        assert (
            1 <= beta < 2
        ), "beta should be in [1,2) for energy score to be strictly proper"

        self.icnn_num_layers = icnn_num_layers
        self.icnn_hidden_size = icnn_hidden_size
        self.is_energy_score = is_energy_score
        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples
        self.estimate_logdet = estimate_logdet

    def create_lightning_module(self) -> MultivariateDeepARLightningModule:
        model = MultivariateDeepARModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=(
                1 + self.num_feat_dynamic_real + len(self.time_features)
            ),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
            icnn_num_layers=self.icnn_num_layers,
            icnn_hidden_size=self.icnn_hidden_size,
            is_energy_score=self.is_energy_score,
            threshold_input = self.threshold_input,
            es_num_samples = self.es_num_samples,
            estimate_logdet=self.estimate_logdet,
        )

        return MultivariateDeepARLightningModule(model=model, loss=self.loss)#, prediction_length=self.prediction_length)