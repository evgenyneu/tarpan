from tarpan.cmdstanpy.psis import (
    save_compare_psis_csv_from_compared,
    save_compare_psis_txt_from_compared,
    save_compare_psis_tree_plot_from_compared,
    compare_psis, save_psis_pareto_k_plot_from_compared,
    ParetoKPlotParams)

from tarpan.cmdstanpy.waic import (
    save_compare_waic_csv_from_compared,
    save_compare_waic_txt_from_compared,
    save_compare_waic_tree_plot_from_compared,
    compare_waic)

from tarpan.cmdstanpy.waic import LPD_COLUMN_NAME_DEFAULT
from tarpan.shared.tree_plot import TreePlotParams
from tarpan.shared.info_path import InfoPath


def save_compare(
        models, lpd_column_name=LPD_COLUMN_NAME_DEFAULT,
        tree_plot_params: TreePlotParams = TreePlotParams(),
        info_path=InfoPath(),
        pareto_k_plot_params: ParetoKPlotParams = ParetoKPlotParams()):
    """
    Compare multiple models using WAIC (Widely Aplicable Information Criterion)
    and PSIS (Pareto-smoothed importance sampling) methods. Saves the analysis
    data and plots.

    Parameters
    ----------

    models : list of dict
        List of model samples from cmdstanpy to compare.

        The dictionary has keys:
            name: str
                Model name
            fit: cmdstanpy.stanfit.CmdStanMCMC
                Contains the samples from cmdstanpy.

    lpd_column_name : str
        Prefix of the columns in Stan's output that contain log
        probability density value for each observation. For example,
        if lpd_column_name='possum', when output is expected to have
        columns 'possum.1', 'possum.2', ..., 'possum.33' given 33 observations.


    info_path : InfoPath
        Determines the location of the output files.
    """

    info_path.set_codefile()

    # Compare with WAIC
    # --------

    compared = compare_waic(models=models, lpd_column_name=lpd_column_name)
    save_compare_waic_csv_from_compared(compared=compared, info_path=info_path)
    save_compare_waic_txt_from_compared(compared=compared, info_path=info_path)

    save_compare_waic_tree_plot_from_compared(
        compared=compared,
        tree_plot_params=tree_plot_params,
        info_path=info_path)

    # Compare with PSIS
    # --------

    compared = compare_psis(models=models, lpd_column_name=lpd_column_name)
    save_compare_psis_csv_from_compared(compared=compared, info_path=info_path)
    save_compare_psis_txt_from_compared(compared=compared, info_path=info_path)

    save_compare_psis_tree_plot_from_compared(
        compared=compared,
        tree_plot_params=tree_plot_params,
        info_path=info_path)

    save_psis_pareto_k_plot_from_compared(
        compared=compared,
        info_path=info_path,
        pareto_k_plot_params=pareto_k_plot_params)
