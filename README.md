# A Python library for analysing cmdstanpy output

This is a collection of functions for analysing output of [cmdstanpy](https://github.com/stan-dev/cmdstanpy) library. The main idea is to do a quick data analysis by calling a single function that makes:

* traceplots of samples,

* text and plots of the summaries of model parameters,

* histograms and pair plots of posterior distributions of parameters.


<img src='images/tarpan.jpg' alt='Picture of Tarpan'>

*The only known illustration of a tarpan made from life, depicting a five month old colt (Borisov, 1841). Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Tarpan.png).*


## Setup

First, install [cmdstanpy library](https://cmdstanpy.readthedocs.io/en/latest/index.html), and then do:

```
pip install tarpan
```

## Usage

### Complete analysis: `save_analysis`

This is the main function of the library that saves summaries
and trace/pair/tree plots in
[model_info](docs/examples/analyse/a01_simple/model_info/analyse) directory.

```Python
from tarpan.cmdstanpy.analyse import save_analysis
model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)
save_analysis(fit, param_names=['mu', 'sigma'])
```

* [Full example code](docs/examples/analyse/a01_simple)

* [Description of output files](docs/save_analysis/output)


### Tree plot: `save_tree_plot`

Function `save_tree_plot` creates a [tree plot](docs/examples/save_tree_plot/a01_single_fit/model_info/tree_plot/summary.pdf).

```Python
from tarpan.cmdstanpy.tree_plot import save_tree_plot
model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)
save_tree_plot([fit], param_names=['mu', 'sigma'])
```

* [Full example code](docs/examples/save_tree_plot/a01_single_fit)

<img src="docs/examples/save_tree_plot/a01_single_fit/model_info/tree_plot/summary.png" width="500" alt="Tree plot">

The two error bars indicate 68% and 95% HPDIs (highest posterior density intervals).


#### Comparing multiple models on a tree plot

Supply multiple fits in order to compare parameters from multiple models.

```Python
from tarpan.cmdstanpy.tree_plot import save_tree_plot
from tarpan.shared.tree_plot import TreePlotParams

# Sample from two models
model1 = CmdStanModel(stan_file="your_model1.stan")
fit1 = model1.sample(data=your_data)
model2 = CmdStanModel(stan_file="your_model2.stan")
fit2 = model2.sample(data=your_data)

# Supply legend labels (optional)
tree_params = TreePlotParams()
tree_params.labels = ["Model 1", "Model 2", "Exact"]
data = [{ "mu": 2.2, "tau": 1.3 }]  # Add extra markers (optional)

save_tree_plot([fit1, fit2], extra_values=data, param_names=['mu', 'tau'],
               tree_params=tree_params)
```

* [Full example code](docs/examples/save_tree_plot/a02_compare_fits)

<img src="docs/examples/save_tree_plot/a02_compare_fits/model_info/tree_plot_compare/summary.png" width="500" alt="Tree plot with multiple models">



### Pair plot: `save_pair_plot`

Function `save_pair_plot` creates a [tree plot](docs/examples/save_tree_plot/a01_single_fit/model_info/tree_plot/summary.pdf). It helps to see correlations between parameters and spot funnel
shaped distributions that can result in sampling problems.

```Python
from tarpan.cmdstanpy.pair_plot import save_pair_plot
model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)
save_pair_plot(fit, param_names=['mu', 'tau', 'eta.1'])
```

* [Full example code](docs/examples/save_pair_plot/a01_pair_plot)

<img src="docs/examples/save_pair_plot/a01_pair_plot/model_info/pair_plot/pair_plot.jpg" width="500" alt="Pair plot">




## Run unit tests

```
pytest
```


## The unlicense

This work is in [public domain](LICENSE).


## 🐴🐴🐴

This work is dedicated to [Tarpan](https://en.wikipedia.org/wiki/Tarpan), an extinct subspecies of wild horse.
