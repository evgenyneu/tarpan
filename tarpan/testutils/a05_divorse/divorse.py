# Generate sampling output, to avoid running in unit tests

from cmdstanpy import CmdStanModel
import pandas as pd
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath


def get_data1_divorse_age():
    #
    # The data contains:
    #   * A: median age at marriage, standardised
    #   * M: 2009 marriage rate per 1000 adults, standardised
    #   * D: 2009 divorce rate per 1000 adults, standardised
    #
    # The data is borrowed from Statistical Rethinking textbook
    # by Richard McElreath.
    #
    # The origin of the data:
    #
    # 1860 census data from http://mapserver.lib.virginia.edu. Marriage
    # and divorce rates from 2009 American Community Survey (ACS).
    # Waffle House density data from wafflehouse.com (retrieved January
    # 2012).
    #
    csv_path = "tarpan/testutils/a05_divorse/divorse.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "D": df["D"].tolist(),
        "A": df["A"].tolist()
    }


def get_data2_divorse_marriage():
    csv_path = "tarpan/testutils/a05_divorse/divorse.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "D": df["D"].tolist(),
        "M": df["M"].tolist()
    }


def get_data3_divorse_age_marriage():
    csv_path = "tarpan/testutils/a05_divorse/divorse.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "D": df["D"].tolist(),
        "A": df["A"].tolist(),
        "M": df["M"].tolist()
    }


def run_model1_divorse_age(data, output_dir, sampling_iters, warmup_iters):
    model_path = "tarpan/testutils/a05_divorse/stan_model/divorse1_divorse_age.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=sampling_iters,
                        warmup_iters=warmup_iters,
                        output_dir=output_dir,
                        seed=1)


def run_model2_divorse_marriage(data, output_dir, sampling_iters,
                                warmup_iters):
    model_path = "tarpan/testutils/a05_divorse/stan_model/divorse2_divorse_marriage.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=sampling_iters,
                        warmup_iters=warmup_iters,
                        output_dir=output_dir,
                        seed=1)


def run_model3_divorse_age_marriage(data, output_dir, sampling_iters,
                                    warmup_iters):
    model_path = "tarpan/testutils/a05_divorse/stan_model/divorse3_divorse_age_marriage.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=sampling_iters,
                        warmup_iters=warmup_iters,
                        output_dir=output_dir,
                        seed=1)


def get_iters():
    return {"sampling_iters": 1000, "warmup_iters": 500}


def get_fit1_divorse_age():
    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a05_divorse1_divorse_age",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    iters = get_iters()
    data = get_data1_divorse_age()

    return run(info_path=info_path, func=run_model1_divorse_age, data=data,
               sampling_iters=iters["sampling_iters"],
               warmup_iters=iters["warmup_iters"])


def get_fit2_divorse_marriage():
    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a05_divorse2_divorse_marriage",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    iters = get_iters()
    data = get_data2_divorse_marriage()

    return run(info_path=info_path, func=run_model2_divorse_marriage,
               data=data,
               sampling_iters=iters["sampling_iters"],
               warmup_iters=iters["warmup_iters"])


def get_fit3_divorse_age_marriage():
    info_path = InfoPath(
                    path='temp_data',
                    dir_name="a05_divorse3_divorse_age_marriage",
                    sub_dir_name=InfoPath.DO_NOT_CREATE
                )

    iters = get_iters()
    data = get_data3_divorse_age_marriage()

    return run(info_path=info_path, func=run_model3_divorse_age_marriage,
               data=data,
               sampling_iters=iters["sampling_iters"],
               warmup_iters=iters["warmup_iters"])
