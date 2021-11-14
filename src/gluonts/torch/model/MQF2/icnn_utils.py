from icnn_lib.flows import SequentialFlow, DeepConvexFlow, ActNorm
from icnn_lib.icnn import (
    PosLinear,
    PosLinear2,
    PICNNAbstractClass,
    ActNormNoLogdet,
    Softplus,
    symm_softplus,
    softplus,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class PICNN(PICNNAbstractClass):
    def __init__(self, dim=2, dimh=16, dimc=2, num_hidden_layers=2, PosLin=PosLinear,
                 symm_act_first=False, softplus_type='gaussian_softplus', zero_softplus=False,
                 is_energy_score=True):
        super(PICNN, self).__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first
        weight_transform = nn.Identity() if is_energy_score else spectral_norm

        # data path
        Wzs = list()
        Wzs.append(weight_transform(nn.Linear(dim, dimh)))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(weight_transform(nn.Linear(dim, dimh)))
        Wxs.append(weight_transform(nn.Linear(dim, 1, bias=False)))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = list()
        Wcs.append(weight_transform(nn.Linear(dimc, dimh)))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for _ in range(num_hidden_layers - 1):
            Wczs.append(weight_transform(nn.Linear(dimh, dimh)))
        Wczs.append(weight_transform(nn.Linear(dimh, dimh, bias=True)))
        self.Wczs = torch.nn.ModuleList(Wczs)

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(weight_transform(nn.Linear(dimh, dim)))
        Wcxs.append(weight_transform(nn.Linear(dimh, dim, bias=True)))
        self.Wcxs = torch.nn.ModuleList(Wcxs)

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(weight_transform(nn.Linear(dimh, dimh)))
        self.Wccs = torch.nn.ModuleList(Wccs)

        # self.actnorm0 = ActNormNoLogdet(dimh)
        # actnorms = list()
        # for _ in range(num_hidden_layers - 1):
        #     actnorms.append(ActNormNoLogdet(dimh))
        # actnorms.append(ActNormNoLogdet(1))
        # self.actnorms = torch.nn.ModuleList(actnorms)
        #
        # self.actnormc = ActNormNoLogdet(dimh)

        self.actnorm0 = nn.Identity() if is_energy_score else ActNormNoLogdet(dimh)
        # actnorms = list()
        # for _ in range(num_hidden_layers - 1):
        #     actnorms.append(ActNormNoLogdet(dimh))

        actnorms = (
            [nn.Identity() if is_energy_score else ActNormNoLogdet(dimh) for _ in range(num_hidden_layers - 1)]
        )

        actnorms.append(nn.Identity() if is_energy_score else ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        self.actnormc = nn.Identity() if is_energy_score else ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))

        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        for Wz, Wx, Wcz, Wcx, Wcc, actnorm in zip(
                self.Wzs[1:-1], self.Wxs[:-1],
                self.Wczs[:-1], self.Wcxs[:-1], self.Wccs,
                self.actnorms[:-1]):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + Wcc(c)))

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](
            self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx)
        )

class CPFlow(DeepConvexFlow):
    def __init__(self, icnn, dim, unbiased=False, no_bruteforce=True, m1=10, m2=None, rtol=0.0, atol=1e-3,
                 bias_w1=0.0, trainable_w0=True, is_energy_score=False, estimate_logdet=False):
        super().__init__(icnn,
                         dim,
                         unbiased=unbiased,
                         no_bruteforce=no_bruteforce,
                         m1=m1,
                         m2=m2,
                         rtol=rtol,
                         atol=atol,
                         bias_w1=bias_w1,
                         trainable_w0=trainable_w0,)
        self.is_energy_score = is_energy_score
        self.estimate_logdet = estimate_logdet

    def get_potential(self, x, context=None):
        n = x.size(0)
        if context is None:
            icnn = self.icnn(x)
        else:
            icnn = self.icnn(x, context)

        if self.is_energy_score:
            return icnn
        else:
            return F.softplus(self.w1) * icnn + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        if self.estimate_logdet:
            return self.forward_transform_stochastic(x, logdet, context=context, extra=extra)
        else:
            return self.forward_transform_bruteforce(x, logdet, context=context)

class SequentialCPFlow(SequentialFlow):
    def __init__(self, flows):
        super().__init__(flows)

    def forward(self, x, context=None):
        for flow in self.flows:
            if isinstance(flow, DeepConvexFlow):
                x = flow.forward(x, context=context)
            else:
                x = flow.forward(x)
        return x

    # def energy_score(self, z, ):