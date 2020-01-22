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

### Doing full analysis with `analyse`

The main method of the library is `analyse` function.

```Python
from tarpan.cmdstanpy import analyse

model = CmdStanModel(stan_file="your_model.stan")
fit = model.sample(data=your_data)

analyse(fit)
```

The purpose of `analyse` is to make full analysis of the sampling output
with a single call and save results in [model_info](docs/examples/analyse/model_info/eight_schools) directory.


#### Example of `analyse`

See the [example code](docs/examples/analyse) of using `analyse`. To run the example,
download [eight_schools.py](docs/examples/analyse/eight_schools.py) and [eight_schools.stan](docs/examples/analyse/eight_schools.stan) into the same directory and run

```
python eight_schools.py
```

The function generates the following files in [model_info/eight_schools](docs/examples/analyse/model_info/eight_schools) directory:


##### [diagnostic.txt](docs/examples/analyse/model_info/eight_schools/diagnostic.txt)

Stan's diagnostic output. Usually, this is the first thing I look at, to see if there were any problems with sampling.


##### [summary.txt](docs/examples/analyse/model_info/eight_schools/summary.txt)

A table showing summaries of distributions for all parameters. The table's format is such that the text can be pasted in Github's Markdown file, like this:

| Name    |   Mean |   Std |   Mode |    + |    - |   68CI- |   68CI+ |   95CI- |   95CI+ |   N_Eff |   R_hat |
|:--------|-------:|------:|-------:|-----:|-----:|--------:|--------:|--------:|--------:|--------:|--------:|
| mu      |   7.88 |  4.90 |   7.09 | 5.46 | 4.23 |    2.85 |   12.54 |   -1.46 |   18.19 |    2438 |    1.00 |
| tau     |   6.58 |  5.64 |   2.16 | 5.72 | 2.16 |    0.00 |    7.88 |    0.00 |   17.44 |    1394 |    1.00 |
| eta.1   |   0.40 |  0.94 |   0.46 | 0.89 | 0.94 |   -0.49 |    1.35 |   -1.46 |    2.32 |    3811 |    1.00 |
| eta.2   |  -0.01 |  0.88 |  -0.05 | 0.84 | 0.86 |   -0.91 |    0.79 |   -1.82 |    1.76 |    4484 |    1.00 |
...



* [summary.pdf](docs/examples/analyse/model_info/eight_schools/summary.pdf): Summary of parameter distributions.

  uses the `fit` output of CmdStanModel's `sample` function and
saves summaries ([text](docs/examples/analyse/model_info/eight_schools/summary.txt), [pdf](docs/examples/analyse/model_info/eight_schools/summary.pdf), and [csv](docs/examples/analyse/model_info/eight_schools/summary.csv)), [traceplots](docs/examples/analyse/model_info/eight_schools/traceplot_01.pdf), [histograms](docs/examples/analyse/model_info/eight_schools/posterior_01.pdf), [diagnostic](docs/examples/analyse/model_info/eight_schools/diagnostic.txt) information to a [model_info](docs/examples/analyse/model_info) directory.





## 🐴🐴🐴

This work is dedicated to [Tarpan](https://en.wikipedia.org/wiki/Tarpan), an extinct subspecies of wild horse.


## The unlicense

This work is in [public domain](LICENSE).
