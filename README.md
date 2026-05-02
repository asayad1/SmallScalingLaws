# Compute-Optimal Tiny Transformers

Replicating the Chinchilla scaling laws at the small scale: sweep over tiny GPT-like models (12K–600K parameters) on WikiText-103 to find the compute-optimal model size and training token count for each FLOP budget, then fit power-law scaling curves.

<table align="center"><tr>
  <td><img src="outputs/plots/scaling/scaling_surface_fixed.png" width="450" /></td>
  <td><img src="outputs/plots/fit_observed_vs_predicted.png" width="450" /></td>
  <td><img src="outputs/plots/scaling/optimal_d_over_n_vs_compute.png" width="450" /></td>
</tr></table>

## Pipeline

1. **Data** - Download & prepare byte-level train/val/test splits from WikiText-103
2. **Sweep** - Generate an iso-compute grid of (model, tokens) pairs across 9 compute budgets
3. **Train** - Train every configuration across multiple seeds and two LR-schedule regimes (matched & fixed-horizon)
4. **Aggregate** - Collect per-run validation losses into a summary table
5. **Fit & Plot** - Fit a Chinchilla-style parametric loss surface and produce scaling-law plots

## Running the Experiment

Everything lives in a single Jupyter notebook:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run all cells top-to-bottom. The notebook will download data, train the sweep, and save plots to `outputs/plots/`.