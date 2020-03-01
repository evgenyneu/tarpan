import os
import shutil

from tarpan.cmdstanpy.compare import save_compare

from tarpan.testutils.a05_divorse.divorse import (
    get_fit1_divorse_age, get_fit2_divorse_marriage,
    get_fit3_divorse_age_marriage)


def test_save_compare():
    fit1_divorse_age = get_fit1_divorse_age()
    fit2_divorse_marriage = get_fit2_divorse_marriage()
    fit3_divorse_age_marriage = get_fit3_divorse_age_marriage()

    models = {
        "Divorse vs Age": fit1_divorse_age,
        "Divorse vs Marriage": fit2_divorse_marriage,
        "Divorse vs Age+Marriage": fit3_divorse_age_marriage
    }

    outdir = "tarpan/cmdstanpy/model_info/compare_test"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    save_compare(models=models)

    # WAIC
    # ------

    assert os.path.isfile(os.path.join(outdir, "compare_waic.csv"))
    assert os.path.isfile(os.path.join(outdir, "compare_waic.txt"))
    assert os.path.isfile(os.path.join(outdir, "compare_waic.pdf"))

    # PSIS
    # ------

    assert os.path.isfile(os.path.join(outdir, "compare_psis.csv"))
    assert os.path.isfile(os.path.join(outdir, "compare_psis.txt"))
    assert os.path.isfile(os.path.join(outdir, "compare_psis.pdf"))

    assert os.path.isfile(os.path.join(outdir,
                          "pareto_k_divorse_vs_age_marriage.pdf"))

    assert os.path.isfile(os.path.join(outdir,
                                       "pareto_k_divorse_vs_age.pdf"))

    assert os.path.isfile(os.path.join(outdir,
                                       "pareto_k_divorse_vs_marriage.pdf"))
