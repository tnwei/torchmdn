"""
Converted to pytorch from https://github.com/tnwei/pure-keras-mdn
"""
import torch


class MDN(torch.nn.Module):
    """
    Defines a Mixture Density Network block comprising of `n_mixtures` of Gaussians.

    Outputs pi, mu, sigma, where:
    + pi: mixture probability, sums to 1
    + mu: means of Gaussians
    + sigma: std dev of Gaussians
    """

    def __init__(self, in_features: int, n_mixtures: int):
        super().__init__()
        self.w_pi = torch.nn.Linear(
            in_features=in_features, out_features=n_mixtures, bias=True
        )
        self.w_mu = torch.nn.Linear(
            in_features=in_features, out_features=n_mixtures, bias=True
        )
        self.w_sigma = torch.nn.Linear(
            in_features=in_features, out_features=n_mixtures, bias=True
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
    Sample `n_points` from the distributions for each point in the test set
    """
    n_test = pi.shape[0]
    result = np.random.random(size=(n_test, n_points))
    noise = np.random.random(size=(n_test, n_points))

    for i in range(n_test):
        for j in range(n_points):
            idx = np.random.choice(a=range(n_mixtures), p=pi[i])
            result[i, j] = mu[i, idx] + noise[i, j] * sigma[i, idx]
    return result
