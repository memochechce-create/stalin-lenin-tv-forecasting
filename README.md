# Lenin and Stalin on Russian television: daily probabilistic forecasts

Course project by **Купча Софья and Пальцева Ксения**. Two people, four channels, two targets: surname mentions and archive records with at least one mention.

## Results and report

- Five-slide presentation: [PowerPoint](output/presentation/lenin_stalin_forecast.pptx) and [PDF](output/presentation/lenin_stalin_forecast.pdf), with speaker notes timed for an eight-minute talk.

- [Visual forecast report](output/joint_forecast/forecast_report.html): self-contained HTML, minimal SVG charts, daily and monthly forecasts, historical evaluation and sensitivity.
- [Final project notebook](output/joint_forecast/final_project.ipynb): explained calculations, model equations, temporal validation, diagnostics and course-method mapping.
- [Methodology, reproduction and file guide](output/joint_forecast/README.md).
- [Daily predictions](output/joint_forecast/forecast_daily.csv), [monthly predictions](output/joint_forecast/forecast_month.csv), [evaluation by series](output/joint_forecast/evaluation_by_series.csv).
- [Original proposal](output/proposal/forecasting_proposal.pdf) and [editable source](output/proposal/forecasting_proposal.tex).

The information cutoff is **2 October 2026**. The proposed 31-day forecast runs **3 October–2 November 2026**. This is a frozen-cutoff retrospective forecast; no observations after the cutoff enter fitting or selection.

## Reproduce from the terminal

Use Python 3.12/3.13 from the repository root:

```powershell
python -m pip install -r requirements-forecast.txt
python analysis/run_joint_forecast.py --jobs 3
python analysis/execute_forecast_notebook.py
python -m unittest discover -s tests -v
```

The workflow fits all models, performs chronological backtesting, produces joint predictive trajectories, validates the results and builds the figures, HTML and notebook. `--jobs 1` runs sequentially. `--rebuild` ignores fitted-posterior caches. The full computation takes time on CPU; valid cached fits are reused automatically. No API credentials or network are needed for forecasting.

## Models and evaluation

All forecasts respect **N → B|N → C=B+E|B**: total archive records, records with a mention, then additional mentions in those records. Future N is forecast rather than observed. N is shared by both people; every simulated path satisfies 0≤B≤N, B≤C and B=0⇒C=0.

1. M1: conjugate constant-rate Poisson/Binomial baseline.
2. M2: negative-binomial and beta-binomial overdispersion, exact static-posterior Metropolis–Hastings.
3. M3: M2 with trend, partially pooled channel/calendar effects and explicit special-date indicators.
4. M4: M3 with random-walk levels and conditional particle filters; an explicitly modular/cut approximation.
5. An equal-weight linear pool of complete trajectories, following the forecast-aggregation lecture.

Calendar/channel prior scales are fixed and disclosed. Four MH chains are required to pass rank-normalized split Rhat<1.01 and bulk/tail ESS≥400. The Bayesian computation is checked separately from model adequacy.

Tuning and model selection use July–December 2024 only. The outer expanding-window backtest covers 21 nonoverlapping calendar months, January 2025–September 2026. Evaluation includes RMSE, CRPS, predictive coverage/width, randomized PIT, horizon-specific scores, block bootstrap uncertainty and origin-block DM tests with Holm correction. Monthly predictive intervals use sums of trajectories, never sums of daily quantiles. Missing days are excluded, not replaced by zero.

## Data and interpretation

The model input is [the corrected daily table](data/cleaned/tv_daily_targets.csv.gz). [Data preparation and checks](output/data_cleaning/README.md) describe coverage, source verification, surname forms, contextual rules, duplicates and repairs. Rebuild it with:

```powershell
python analysis/prepare_clean_data.py
```

Source: [GDELT](https://gdeltproject.org/) television transcripts and archived metadata, January 2023–October 2026. The source collection workflow is in `data_collection/`. Original transcripts/SQLite and rebuildable caches remain local and are ignored by Git. The supplied compressed daily data are sufficient to reproduce forecasting without Google Cloud access.

Counts include all archived programmes, advertisements and true rebroadcasts. A day is defined by the archive record's MSK start date. Contextual surname labels remain provisional; independent human precision/recall is not established. A separate expanded-context sensitivity forecast is supplied. Archive volume and ASR/text length can affect counts: the forecasts support statements about the measured archive series, not causal claims about political direction.

The cancelled neural-disambiguation experiment is documented [here](output/data_cleaning/neural/EXPERIMENT_NOTES.md); it does not affect the active rules or forecasts.

## Historical pilot

`output/stalin_russia1/`, `analysis/stalin_models.py` and the old `requirements.txt` reproduce an earlier Stalin/Russia 1 pilot on unfiltered candidates and the former coverage definition. These files are retained for provenance. Use the new `joint_forecast` results and `requirements-forecast.txt` for the final project.
