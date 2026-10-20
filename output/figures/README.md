# Final-model figures

The active plot generator is `analysis/project_figures.py`. Final model: M3HR.
All saved labels use ordinary hyphens instead of long dash characters (U+2012 through U+2015).

New outputs:
- `daily_total_mentions`: daily number of mentions, sum of all four channels, for Lenin and Stalin.
- `daily_total_broadcasts`: daily number of archive records with a mention, sum of all four channels.

Aggregate intervals come from sums within each joint predictive draw, not sums of channel quantiles. Historical totals require all four channels to be observed; incomplete days remain gaps.

Generate the final predictive paths if absent:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python analysis/run_forecast_v2.py --phase final --models M3HR
```

Generate selected English figures (without rerunning the backtest):

```powershell
python analysis/project_figures.py --lang en --output-dir output/figures/en --figures daily_total_mentions,daily_total_broadcasts,daily_mentions,daily_broadcasts
```

For headerless presentation images, replace `--output-dir` with `--slides-dir output/figures/slides_en`.

The complete `--figures all` run also needs the original per-origin test caches. The supplied older SVGs are cached plotting outputs; their labels can be refreshed without changing chart geometry, counts, scores or intervals:

```powershell
python analysis/refresh_saved_figure_labels.py --source-dir output/figures --output-dir output/figures/en --lang en
node analysis/render_saved_figures.mjs output/figures/en
```

The last command uses the bundled Node runtime and requires `RUNTIME_NODE_MODULES` to point to its packages. English translations are in `analysis/figure_label_translations.json`. Russian saved outputs can be refreshed with `--lang ru`.

Checks: `python -m unittest discover -s tests -p test_project_figures.py -v`.
