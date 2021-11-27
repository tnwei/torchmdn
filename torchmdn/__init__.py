"""
Converted to pytorch from https://github.com/tnwei/pure-keras-mdn
"""
import torch

class MDN(torch.nn.Module):
    def __init__(self, in_features: int, n_mixtures: int):
        super().__init__()
        self.w_pi = torch.nn.Linear(in_features=in_features, out_features=n_mixtures, bias=True)
        self.w_mu = torch.nn.Linear(in_features=in_features, out_features=n_mixtures, bias=True)
        self.w_sigma = torch.nn.Linear(in_features=in_features, out_features=n_mixtures, bias=True)
        
    def forward(self, x):
        pi = self.w_pi(x)
        pi = torch.nn.functional.softmax(pi, dim=1) # C of NC
        mu = self.w_mu(x)
        sigma = self.w_sigma(x)
        sigma = torch.exp(sigma)

        return pi, mu, sigma
    
def gaussian_pdf(y, mu, sigma):
    # Output should be in the shape of (n_mixtures, )
    result = torch.square(y - mu) / torch.square(sigma) * -0.5
    result = torch.exp(result) / torch.sqrt(2*np.pi*torch.square(sigma))
    return result

def mdn_likelihood_loss(y, pi, mu, sigma):
    loss = gaussian_pdf(y, mu, sigma) * pi
    loss = torch.sum(loss, axis=1)
    loss = - torch.log(loss + 1e-10) # Prevent NaN
    loss = torch.mean(loss, axis=0)
    return loss