# torchmdn (WIP)

A simple module for Mixture Density Networks in Pytorch. 

## Example

Install with `pip install git+https://github.com/tnwei/torchmdn`

``` python
from torchmdn import MDN, mdn_likelihood_loss, sample_mdn
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Generate sample data
def sample_fun(x):
    return 7 * np.sin(0.75*x) + 0.5*x +\
    np.random.normal(loc=0, scale=1, size=np.squeeze(x).shape)

ys = np.linspace(-10, 10, num=2000)
xs = np.apply_along_axis(sample_fun, axis=0, arr=ys)
xs = xs.reshape(-1, 1)
ys = ys.reshape(-1, 1)

# X_train.shape, y_train.shape = (1600, 1)
# X_test.shape, y_test.shape = (400, 1)
X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)

# Convert to torch tensors
X_train_ten = torch.FloatTensor(X_train)
X_test_ten = torch.FloatTensor(X_test)
y_train_ten = torch.FloatTensor(y_train)
y_test_ten = torch.FloatTensor(y_test)

# Create simple NN: 1 Dense layer -> MDN with 5 mixtures
class Net(torch.nn.Module):
    def __init__(self, n_mixtures):
        super().__init__()
        self.w = torch.nn.Linear(in_features=1, out_features=20, bias=True)
        self.mdn = MDN(in_features=20, n_mixtures=n_mixtures)
        
    def forward(self, x):
        out = self.w(x)
        out = torch.tanh(out)
        out = self.mdn(out)
        return out
   
n_mixtures = 5
net = Net(n_mixtures=5)
opt = torch.optim.Adam(net.parameters())

# Train simple NN
for i in range(10000):
    opt.zero_grad()
    pi, mu, sigma = net(X_train_ten)
    loss = mdn_likelihood_loss(y_train_ten, pi, mu, sigma)
    loss.backward()
    opt.step()
    
    if i % 500 == 0:
        print(loss.item())

# Sample 10 points from the learned distributions
pi, mu, sigma = net(X_test_ten)
pi, mu, sigma = pi.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()
y_pred = sample_mdn(pi, mu, sigma, 10) # y_pred.shape = (400, 10)
```
