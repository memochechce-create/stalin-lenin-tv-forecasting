# Lenin and Stalin on Russian television

Course project by **Sofya Kupcha, Ksenia Paltseva and Timofey Mingalev**.

## Final deliverables

- [Final seven-slide English presentation](output/presentation/lenin_stalin_forecast_final.pptx). Slide 5 shows daily forecasts by channel; slide 6 shows the monthly-total forecast; slide 7 shows daily totals across all four channels for both targets.
- [Final M3HR model, evaluation and numerical results](output/forecast_v2/README.md).
- [English figures](output/figures/en), [Russian figures and reproduction guide](output/figures/README.md).
- [Monthly forecasts](output/forecast_v2/forecast_month_v2.csv), [event probabilities](output/forecast_v2/forecast_events_v2.csv), [selection record and test metrics](output/forecast_v2/decisions.json).
- [Corrected daily input](data/cleaned/tv_daily_targets.csv.gz) and [data preparation](output/data_cleaning/README.md).

The targets are daily counts of person mentions and archive records with at least one mention: two people, four channels, two targets, sixteen series. The forecast horizon is **3 October - 2 November 2026**, with information through **2 October 2026**. Missing archive days are excluded, not treated as zeros.

## Model and evaluation

The final model is **M3HR**: the calendar-and-trend model M3 with an estimated shrinkage scale for month-of-year effects and a calendar-month random effect shared across channels and days within each component. The model simulates archive volume N, records with mentions B conditional on N, and all mentions C=B+E. Joint paths respect the count constraints. Inference uses Hamiltonian Monte Carlo with four chains, R-hat below 1.01, bulk/tail ESS at least 400 and no divergences.

M1 is the constant-rate baseline; M2 adds overdispersion; M3 adds calendar effects and trend; M4 adds dynamic random-walk levels. The retained [baseline implementation and results](output/joint_forecast/README.md) are dependencies and comparison material, not an alternative final deliverable.

The original test has 21 expanding-window months, January 2025 - September 2026. In the colleague's within-run comparison, final-model daily scaled CRPS is 0.453729 versus M3's 0.453936; monthly scaled CRPS is 0.190925 versus 0.196740. Monthly 80%/95% coverage is 82.9%/94.3%. The final structure was selected after inspection of the test, so incremental gains remain exploratory; see the saved selection record.

## Reproduce

```powershell
python -m pip install -r requirements-forecast-v2.txt
$env:OPENBLAS_NUM_THREADS='1'
python analysis/run_forecast_v2.py --phase final --models M3HR
python analysis/project_figures.py --lang en --output-dir output/figures/en --figures daily_total_mentions,daily_total_broadcasts,daily_mentions,daily_broadcasts,calendar_effects
python -m unittest discover -s tests -v
```

To reconstruct the entire backtest, follow [the final-model guide](output/forecast_v2/README.md). This is a longer computation. Rebuildable posterior/test caches are not versioned. The small final predictive trajectory file is included so the final daily figures can be regenerated without refitting. `analysis/refresh_final_presentation.mjs` refreshes the embedded figures while preserving the final deck's layout and order.

All figures are produced by project code. Four-channel intervals are computed after summing channels within each joint draw. Long dash characters are normalized before saving. Final refit means differ from the colleague's supplied monthly summary by less than 0.075%; [the refit comparison](output/forecast_v2/final_refit_comparison.json) records the small sampling/platform differences.

Raw television transcripts and collection databases remain local and ignored. Collection, cleaning, source audit and corrected-input validation are retained for provenance. Older pilot presentations and the cancelled neural experiment have been removed from the current tree; Git history retains the previously committed versions.
