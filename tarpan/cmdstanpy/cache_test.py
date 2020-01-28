import shutil
import os
from tarpan.cmdstanpy.cache import run
from cmdstanpy import CmdStanModel


def myfunc(output_dir, data):
    data["count"] += 1

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    model_path = "tarpan/testutils/a01_eight_schools/eight_schools.stan"
    model = CmdStanModel(stan_file=model_path)

    return model.sample(data=data, chains=1, cores=1,
                        sampling_iters=1000, warmup_iters=1000,
                        output_dir=output_dir)


def test_run():

    outdir = "tarpan/cmdstanpy/model_info/cache_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    data = {"count": 1}

    fit = run(myfunc, data=data)

    assert os.path.isfile(os.path.join(outdir, "stan_cache", "fit.pkl"))
    assert 'tau' in fit.column_names
    assert fit.get_drawset()['tau'].shape == (1000,)

    # Call it again
    fit = run(myfunc, data=data)

    assert 'tau' in fit.column_names
    assert fit.get_drawset()['tau'].shape == (1000,)

    # Ensure `myfunc` was called once
    assert data["count"] == 2
