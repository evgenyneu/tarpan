# Tools for cmdstanpy

This is a collection of functions for analysing output of [cmdstanpy](https://github.com/stan-dev/cmdstanpy) library, written in Python.

```Python
from tarpan.cmdstanpy import analyse

model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)
analyse(fit)  # Save summaries, traceplots and histograms
```

Tarpan's includes functions for making:

* traceplots of samples,

* text and plots of the summaries of model parameters,

* histograms of posterior distributions of parameters.


<img src='images/tarpan.jpg' alt='Picture of Tarpan'>

*The only known illustration of a tarpan made from life, depicting a five month old colt (Borisov, 1841). Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Tarpan.png).*


## Setup

```
pip install tarpan
```


## 🐴🐴🐴

This work is dedicated to [Tarpan](https://en.wikipedia.org/wiki/Tarpan), an extinct subspecies of wild horse.


## The unlicense

This work is in [public domain](LICENSE).
