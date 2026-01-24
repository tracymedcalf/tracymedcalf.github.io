---
title: "Deep Energy-Based Model Fitted with Noise Contrastive Estimation"
date: "2026-01-17"
updated: "2026-01-24"
categories:
  - "sveltekit"
  - "markdown"
  - "svelte"
coverImage: "/images/wine.jpg"
coverWidth: 4
coverHeight: 5
excerpt: ...my attempt to better understand the process of training an Energy-Based Model.
---
# Deep Energy-Based Model Fitted with Noise Contrastive Estimation (NCE)

This notebook is my attempt to better understand the process of training an Energy-Based Model.

The model is fitted to tabular data (the wine quality data set from the UCI repository). Credit is due to volagold (https://github.com/volagold/nce/), without whose code example I would not have been able to write this.


```python
import math

import torch
from torch import nn, optim
import pandas as pd
import numpy as np

# device selection for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```


```python
# the number of categorical and numerical columns
df = pd.read_csv("./wine_quality/winequality-red.csv", delimiter=";")
num_features = len(df.columns)
batch_size = df.shape[0]
X = torch.tensor(df.values, dtype=torch.float32).to(device)
```

Define a feed-forward neural network to output the energy score.

From the PyTorch docs:

`Parameter` is a "kind of Tensor that is to be considered a module parameter".

That is, a Parameter is automatically included in the `parameters()` iterator.

In NCE, $\log_Z(\theta)$ is treated as a learnable parameter (Song et al., 2021).


```python
class FeedForwardNN(nn.Module):
    
    def __init__(self, dims=32):
        super(FeedForwardNN, self).__init__()
        self.log_Z_of_theta = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.f = nn.Sequential(
            nn.Linear(num_features, dims),
            nn.LeakyReLU(0.2),
            nn.Linear(dims, dims),
            nn.LeakyReLU(0.2),
            nn.Linear(dims, 1)
        )
        
    def forward(self, x):
        return -self.f(x) - self.log_Z_of_theta
```

From the PyTorch docs, `MultivariateNormal` creates "a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix".

From the PyTorch docs, `eye` returns "a 2-D tensor with ones on the diagonal and zeros elsewhere".

This code will return a Gaussian with dims equal to `num_features`, a mean at 0, and a covariance matrix of `I`.


```python
model = FeedForwardNN().to(device)
num_rows = num_features
noise = torch.distributions.MultivariateNormal(torch.zeros(num_rows, device=device), torch.eye(num_rows, device=device))
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
```


```python
MAX_EPOCHS = 1000
for i in range(MAX_EPOCHS):
     
    optimizer.zero_grad()
    

    # GENERATE NOISE
    gen = noise.sample((len(X),))
    
    
    # CALCULATE THE ENERGY LOSS
    logp_x = model(X)
    logq_x = noise.log_prob(X).unsqueeze(1)
    logp_gen = model(gen)
    logq_gen = noise.log_prob(gen).unsqueeze(1)
    
    value_data = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)
    value_gen = logq_gen - torch.logsumexp(torch.cat([logp_gen, logq_gen], dim=1), dim=1, keepdim=True)
    
    loss = -(value_data.mean() + value_gen.mean())
    
    with torch.no_grad():
        r_x = torch.sigmoid(logp_x - logq_x)
        r_gen = torch.sigmoid(logq_gen - logp_gen)
        acc = ((r_x > 0.5).float().mean() + (r_gen > 0.5).float().mean()) / 2
    
    
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print("Loss:", loss.item())
```


```python
x = torch.randn(num_features, device=device, requires_grad=True)
optimizer = optim.Adam([x])
model(x)
```

The above, randomly generated vector has a higher energy score than the vectors below that are the result of optimization. That's appropriate. The higher the energy score, the more out-of-distribution it is.


```python
STEPS = 100
for _ in range(STEPS):
    optimizer.zero_grad()
    energy = model(x)
    energy.backward()
    optimizer.step()
    
x_star = x.detach()
x_star
```


```python
example = X[10].detach().clone()
example[-1] = 9.0
example
```


```python
w_opt = torch.nn.Parameter(example[-1].detach().clone())
optimizer = optim.Adam([w_opt])


for _ in range(1000):
    optimizer.zero_grad()
    ex = example.clone()
    ex[-1] = w_opt
    energy = model(ex)
    energy.backward()
    optimizer.step()
    
example[-1] = w_opt
example
```

The true rating given to the above data point was 5. The model estimates the rating to be 10.

Clearly, this model is not ready to generate plausible data points. More work is needed.

One last thing: for our amusement, let's show that the energy score is higher when the rating is lower for this data point.


```python
a = example.clone()
a[-1] = 9.0
(model(a), model(example))
```
