# Tools for cmdstanpy

This is a collection of functions for analysing output of [cmdstanpy](https://github.com/stan-dev/cmdstanpy) library, written in Python.

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

## Usage

### `save_analysis`

This is the main function of the library that makes full analysis of
Stan's output samples saves results in
[model_info](docs/examples/analyse/a01_simple/model_info/analyse) directory.

```Python
from tarpan.cmdstanpy.analyse import save_analysis

model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)

save_analysis(fit)  # <-- Creates analysis files in `model_info` directory
```


### Example of `save_analysis`

See the [example code](docs/examples/analyse/a01_simple) of using `analyse`. To run the example,
download analyse.py and eight_schools.stan files into the same directory and run

```
python analyse.py
```

The function generates the following files in [model_info/analyse](docs/examples/analyse/a01_simple/model_info/analyse) directory:


#### 1. Model's diagnostic info [diagnostic.txt](docs/examples/analyse/a01_simple/model_info/analyse/diagnostic.txt)

Stan's diagnostic output. Usually, this is the first thing I look at, to see if there were any problems with sampling.


#### 2. Text summary [summary.txt](docs/examples/analyse/a01_simple/model_info/analyse/summary.txt)

A table showing summaries of distributions for all parameters. The table's format is such that the text can be pasted in Github's Markdown file, like this:

| Name    |   Mean |   Std |   Mode |    + |    - |   68CI- |   68CI+ |   95CI- |   95CI+ |   N_Eff |   R_hat |
|:--------|-------:|------:|-------:|-----:|-----:|--------:|--------:|--------:|--------:|--------:|--------:|
| mu      |   7.88 |  4.90 |   7.09 | 5.46 | 4.23 |    2.85 |   12.54 |   -1.46 |   18.19 |    2438 |    1.00 |
| tau     |   6.58 |  5.64 |   2.16 | 5.72 | 2.16 |    0.00 |    7.88 |    0.00 |   17.44 |    1394 |    1.00 |
| eta.1   |   0.40 |  0.94 |   0.46 | 0.89 | 0.94 |   -0.49 |    1.35 |   -1.46 |    2.32 |    3811 |    1.00 |
| eta.2   |  -0.01 |  0.88 |  -0.05 | 0.84 | 0.86 |   -0.91 |    0.79 |   -1.82 |    1.76 |    4484 |    1.00 |

The summary columns are:

*  **Name, Mean, Std** are the name of the parameter, its mean and standard deviation.

*  **68CI-, 68CI+, 95CI-, 95CI+** are the 68% and 95% HPDIs (highest probability density intervals).

* **Mode, +, -** is a mode of distribution with upper and lower uncertainties, which are calculated as distances to 68% HPDI.

* **N_Eff** is Stan's number of effective samples, the higher the better.

* **R_hat** is a Stan's parameter representing the quality of the sampling. We need this to be near 1.


#### 3. [summary.csv](docs/examples/analyse/a01_simple/model_info/analyse/summary.csv)

Same as summary.txt but in CSV format.


#### 4. Summary tree plot [summary.pdf](docs/examples/analyse/a01_simple/model_info/analyse/summary.pdf)

The plot of the summary:

<img src="docs/examples/analyse/a01_simple/model_info/analyse/summary.png" width="700" alt="Summary plot">



* [summary.pdf](docs/examples/analyse/model_info/eight_schools/summary.pdf): Summary of parameter distributions.

  uses the `fit` output of CmdStanModel's `sample` function and
saves summaries ([text](docs/examples/analyse/model_info/eight_schools/summary.txt), [pdf](docs/examples/analyse/model_info/eight_schools/summary.pdf), and [csv](docs/examples/analyse/model_info/eight_schools/summary.csv)), [traceplots](docs/examples/analyse/model_info/eight_schools/traceplot_01.pdf), [histograms](docs/examples/analyse/model_info/eight_schools/posterior_01.pdf), [diagnostic](docs/examples/analyse/model_info/eight_schools/diagnostic.txt) information to a [model_info](docs/examples/analyse/model_info) directory.


## Run unit tests

```
pytest
```


## 🐴🐴🐴

This work is dedicated to [Tarpan](https://en.wikipedia.org/wiki/Tarpan), an extinct subspecies of wild horse.


## The unlicense

This work is in [public domain](LICENSE).
