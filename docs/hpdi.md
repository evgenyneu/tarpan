# How to change the widths of HPD intervals.

By default, Tarpan uses 68.27% and 95.45% HPDIs (highest posterior density intervals).
If you want to change these values, you can supply `summary_params` parameter to
any Tarpan's function. For example, here we create a summary with 5% and 99% HPDIs:

```Python
from tarpan.shared.summary import SummaryParams
save_summary(fit, summary_params=SummaryParams(hpdis=[0.05, 0.99]))
```

See [example code here](/docs/examples/save_summary/a02_save_summary_configure_hpdi).
