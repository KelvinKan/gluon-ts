import torch
import numpy as np

Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z

def log_normal_2D(x, mean, covar):
    z = -float(np.log(2 * np.pi))
    logdet_term = -0.5 * np.log(np.linalg.det(covar))
    covarinv = np.linalg.inv(covar)
    covarinv = torch.tensor(np.tile(covarinv, [x.shape[0],1,1]))
    mean = torch.tensor(mean)
    xMmean = (x-mean).unsqueeze(-1)
    matmul_term = -0.5*torch.matmul(torch.transpose(xMmean, -2, -1),
                                    torch.matmul(covarinv, xMmean))
    matmul_term = matmul_term.squeeze(-1).squeeze(-1)
    # matmul_term = -0.5*np.matmul(np.transpose(xMmean,(0,2,1)), np.matmul(covarinv, xMmean))
    # matmul_term = np.squeeze(np.squeeze(matmul_term, -2), -1)

    return matmul_term + z + logdet_term

def log_standard_normal(x):
    z = - 0.5 * Log2PI
    return - x ** 2 / 2 + z

# mean = np.array([1,1])
# covar = np.array([[1,-.5],[-.5, 1]])
# x = torch.rand([2,2])
#
# print(log_normal_2D(x, mean, covar))