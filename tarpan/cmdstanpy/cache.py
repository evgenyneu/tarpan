import os
import pickle
from tarpan.shared.info_path import InfoPath, get_info_dir


def run(func, info_path=InfoPath(), **kwargs):
    """
    A function that helps to run Stan only once and return cached result
    next time instead.

    Parameters
    ----------
    fun: function
        A function that initialises CmdStanModel() model, runs
        model.sample(...) and returns its result (cmdstanpy.CmdStanMCMC object).
        The function must have an `output_dir` parameter, which should be
        pass to model.sample:
            model.sample(output_dir=output_dir)
        for storing the stan's CSV files in correct location.

    **kwargs: any other argument that will be passed to `func`

    Returns
    -------
    mdstanpy.CmdStanMCMC
        Contains the samples from cmdstanpy.
    """

    info_path.set_codefile()
    fit_dir = os.path.join(get_info_dir(info_path), "stan_cache")
    fit_path = os.path.join(fit_dir, "fit.pkl")

    if not os.path.exists(fit_path):
        os.makedirs(fit_dir, exist_ok=True)
        fit = func(**kwargs, output_dir=fit_dir)

        # Load samples into the fit object, so they can be retrived faster
        temp = fit.get_drawset()

        with open(fit_path, 'wb') as file:
            pickle.dump(fit, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fit_path, 'rb') as input:
        return pickle.load(input)
