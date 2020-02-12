"""Save a text summary comparing parameters from different models"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.summary import SummaryParams
from tarpan.shared.param_names import filter_param_names

from tarpan.shared.compare_parameters import (
    CompareParametersType,
    save_compare_parameters as shared_save_compare_parameters)


def save_compare_parameters(
        fits,
        labels,
        param_names=None,
        type: CompareParametersType = CompareParametersType.TEXT,
        info_path=InfoPath(),
        summary_params=SummaryParams()):
    """
    Saves a text table that compares model parameters

    Parameters
    ----------

    fits : list of cmdstanpy.stanfit.CmdStanMCMC

        Contains the samples from cmdstanpy.

    labels : list of str

        Names of the models in `fits` list.

    param_names : list of str

        Names of parameters. Include all if None.

    type : CompareParametersType

        Format of values in the text table.

    info_path : InfoPath

        Path information for creating summaries.

    """

    info_path.set_codefile()
    models = []

    for fit in fits:
        param_names = filter_param_names(fit.column_names, param_names)
        samples = fit.get_drawset(params=param_names)
        models.append(samples)

    shared_save_compare_parameters(models, labels=labels, type=type,
                                   param_names=param_names,
                                   info_path=info_path,
                                   summary_params=summary_params)
