"""
Converted to pytorch from https://github.com/tnwei/pure-keras-mdn
"""
import torch
import numpy as np
from typing import Optional


class MDN(torch.nn.Module):
    """
    Defines a Mixture Density Network block comprising of `n_mixtures` of Gaussians.

    Args
    ----
    + in_features: size of each input sample
    + n_mixtures: number of Gaussian mixtures
    + bias_init: If not None, will be used to initialize the bias. Must have same dims
        as n_mixtures.
            Defaults to None.

    Outputs pi, mu, sigma, where:
    + pi: mixture probability, sums to 1
    + mu: means of Gaussians
    + sigma: std dev of Gaussians
    """

    def __init__(
        self,
        in_features: int,
        n_mixtures: int,
        bias_init: Optional = None,
        device: Optional[str] = None,
        dtype: Optional = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w_pi = torch.nn.Linear(
            in_features=in_features,
            out_features=n_mixtures,
            bias=True,
            **factory_kwargs,
        )
        self.w_mu = torch.nn.Linear(
            in_features=in_features,
            out_features=n_mixtures,
            bias=True,
            **factory_kwargs,
        )

        if bias_init is not None:
            # ref: https://discuss.pytorch.org/t/fix-bias-and-weights-of-a-layer/75120/5
            with torch.no_grad():
                init_bias = torch.FloatTensor(bias_init, device=device)
                assert (
                    init_bias.size == self.w_mu.bias.size
                ), f"init_bias shape is {init_bias.shape}, needs to be {self.w_mu.bias.size}"
                self.w_mu.bias = torch.nn.parameter.Parameter(
                    data=init_bias, requires_grad=True
                )
        else:
            pass

        self.w_sigma = torch.nn.Linear(
            in_features=in_features,
            out_features=n_mixtures,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, x):
        pi = self.w_pi(x)
        pi = torch.nn.functional.softmax(pi, dim=1)  # C of NC
        mu = self.w_mu(x)
        sigma = self.w_sigma(x)
        sigma = torch.exp(sigma)

        return pi, mu, sigma


def gaussian_pdf(y, mu, sigma):
    """
    Used in `mdn_likelihood_loss` to reparametrize MDN output wrt each mixture Gaussian
    """
    # Output should be in the shape of (n_mixtures, )
    result = torch.square(y - mu) / torch.square(sigma) * -0.5
    result = torch.exp(result) / torch.sqrt(2 * np.pi * torch.square(sigma))
    return result


def mdn_likelihood_loss(y, pi, mu, sigma):
    """
    Calculates the likelihood loss between the mixture of Gaussians
    and label `y`
    """
    loss = gaussian_pdf(y, mu, sigma) * pi
    loss = torch.sum(loss, axis=1)
    loss = -torch.log(loss + 1e-10)  # Prevent NaN
    loss = torch.mean(loss, axis=0)
    return loss


def sample_mdn(pi, mu, sigma, n_points=10):
    """
    Sample `n_points` from the mixture of Gaussians
    """
    n_samples, n_mixtures = pi.shape[0], pi.shape[1]
    result = np.random.random(size=(n_samples, n_points))
    noise = np.random.random(size=(n_samples, n_points))

    for i in range(n_samples):
        for j in range(n_points):
            idx = np.random.choice(a=range(n_mixtures), p=pi[i])
            result[i, j] = mu[i, idx] + noise[i, j] * sigma[i, idx]
    return result
