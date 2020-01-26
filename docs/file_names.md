# Where does Tarpan puts summary and plot files?

By default, all files are placed into `model_info/CODEFILE` directory, where `CODEFILE` is the name of your python script file. For example, suppose you called Tarpan's `save_summary` function from a script file called `make_plots.py`. In this case, Tarpan will create a summary file at `model_info/make_lots/summary.txt` location.

Why does Tarpan do this? This is "convention over configuration" approach: Tarpan will choose default file locations itself, so you don't have to worry about the file paths and focus on science. :)


## How to change location and file names of generated files?

In order to change the location of a summary/plot file, supply `info_path`
parameter to any of Tarpan's function. For example, here is how to
create a tree plot at  `~/tarpan/analysis/model1/normal.png` location in
your user's home directory:

```Python
from tarpan.shared.summary import SummaryParams

save_tree_plot([fit],
               info_path=InfoPath(
                    path='~/tarpan',
                    dir_name="analysis",
                    sub_dir_name="model1",
                    base_name="normal",
                    extension="png"
               ))
```

See [more examples here](/docs/examples/save_tree_plot/a03_custom_location/custom_location.py).
