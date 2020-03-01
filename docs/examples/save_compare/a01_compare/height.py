# Example of comparing three models using WAIC and PSIS
#
# Usage
# -----
#   python height.py
#
# Results are saved to model_info directory
from cmdstanpy import CmdStanModel
import pandas as pd
from tarpan.cmdstanpy.compare import save_compare


def get_data1_intercept():
    # The data contains:
    #   * h0: initial height of plant
    #   * h1: final height
    #   * fungus: 0, 1 presence of fungus
    #   * treatment: 0, 1 presence of anti-fungal treatment
    # The data is borrowed from Statistical Rethinking textbook
    # by Richard McElreath. All rights belong to the author
    csv_path = "height.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "h0": df["h0"].tolist(),
        "h1": df["h1"].tolist()
    }


def get_data2_fungus_treatment():
    csv_path = "height.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "h0": df["h0"].tolist(),
        "h1": df["h1"].tolist(),
        "fungus": df["fungus"].tolist(),
        "treatment": df["treatment"].tolist()
    }


def get_data3_treatment():
    csv_path = "height.csv"
    df = pd.read_csv(csv_path)

    return {
        "n": len(df.index),
        "h0": df["h0"].tolist(),
        "h1": df["h1"].tolist(),
        "treatment": df["treatment"].tolist()
    }


def run_model1_intecept(data):
    model_path = "stan_model/height1_intercept.stan"
    model = CmdStanModel(stan_file=model_path)
    return model.sample(data=data, chains=1, cores=1, seed=1)


def run_model2_fungus_treatment(data):
    model_path = "stan_model/height2_fungus_treatment.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1, seed=1)


def run_model3_treatment(data):
    model_path = "stan_model/height3_treatment.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1, seed=1)


def run_model():
    data1 = get_data1_intercept()
    fit1_intercept = run_model1_intecept(data1)

    data2 = get_data2_fungus_treatment()
    fit2_fungus_treatment = run_model2_fungus_treatment(data2)

    data3 = get_data3_treatment()
    fit3_treatment = run_model3_treatment(data3)

    models = {
        "Intercept": fit1_intercept,
        "Fungus+treatment": fit2_fungus_treatment,
        "Treatment": fit3_treatment
    }

    save_compare(models=models, lpd_column_name="lpd_pointwise")


if __name__ == '__main__':
    run_model()
    print('We are done')
