"""
Specify custom location for the summary file
"""

from cmdstanpy import CmdStanModel
from tarpan.cmdstanpy.tree_plot import save_tree_plot
from tarpan.shared.info_path import InfoPath


def run_model():
    model = CmdStanModel(stan_file="eight_schools.stan")

    data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]
    }

    fit = model.sample(data=data, chains=4, cores=4, seed=1,
                       sampling_iters=1000, warmup_iters=1000)

    # Change all path components:
    #   ~/tarpan/analysis/model1/normal.png
    save_tree_plot([fit],
                   info_path=InfoPath(
                        path='~/tarpan',
                        dir_name="analysis",
                        sub_dir_name="model1",
                        base_name="normal",
                        extension="png"
                   ))

    # Change the file name:
    #   model_into/custom_location/my_summary.pdf
    save_tree_plot([fit],
                   info_path=InfoPath(base_name="my_summary"))

    # Change the file type:
    #   model_into/custom_location/summary.png
    save_tree_plot([fit],
                   info_path=InfoPath(extension="png"))

    # Change the sub-directory name:
    #   model_into/custom/summary.pdf
    save_tree_plot([fit],
                   info_path=InfoPath(sub_dir_name="custom"))

    # Do not create sub-directory name:
    #   model_into/summary.pdf
    save_tree_plot([fit],
                   info_path=InfoPath(sub_dir_name=InfoPath.DO_NOT_CREATE))

    # Change the default top directory name from `model_info`:
    #   my_files/custom_location/summary.pdf
    save_tree_plot([fit],
                   info_path=InfoPath(dir_name='my_files'))

    # Change the root path to tarpan in your user directory
    #   ~/tarpan/model_info/custom_location/summary.pdf
    save_tree_plot([fit],
                   info_path=InfoPath(path='~/tarpan'))


if __name__ == '__main__':
    run_model()
    print('We are done')
