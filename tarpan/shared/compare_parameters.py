"""Save a text summary comparing parameters from different models"""

from tarpan.shared.info_path import InfoPath
from tarpan.shared.summary import SummaryParams
from tarpan.shared.param_names import filter_param_names


def save_compare_parameters(models,
                            param_names=None,
                            info_path=InfoPath(),
                            summary_params=SummaryParams()):
    """
    Saves a text table that compares model parameters

    Parameters
    ----------

    models : list Panda's data frames

        List of data frames for each model, containg sample values for
        multiple parameters (one parameter is one data frame column).
        Supply multiple data frames to compare parameter distributions.

    param_names : list of str

        Names of parameters. Include all if None.

    info_path : InfoPath

        Path information for creating summaries.

    """

    info_path.set_codefile()
