from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.cache import run


def run_stan(output_dir, mydata):
    model = CmdStanModel(stan_file="eight_schools.stan")

    return model.sample(data=mydata, chains=4, cores=4, seed=1,
                        sampling_iters=1000, warmup_iters=1000,
                        output_dir=output_dir)  # Make sure to pass this in


def run_model():
    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    # Will only run Stan once and cache the samples to disk
    fit = run(func=run_stan, mydata=data)
    print('\nColumn names: ')
    print(fit.column_names)


if __name__ == '__main__':
    run_model()
    print('We are done')
