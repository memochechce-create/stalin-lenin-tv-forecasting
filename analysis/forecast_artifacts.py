"""Publication-style SVG/PNG figures, standalone HTML and explained notebook.

Rendering is deterministic and local: Matplotlib, no Plotly/CDN/web fonts.
"""
from pathlib import Path
import html
import json
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

import joint_count_models as m

OUT = m.ROOT/'output/joint_forecast'
FIG = OUT/'figures'
COLORS = {'lenin': '#B46A48', 'stalin': '#277978'}
INK = '#243C45'; MUTED = '#748188'; PAPER = '#FCFBF8'; GRID = '#E7E8E4'
LABELS = {'1TV': 'Первый канал', 'NTV': 'НТВ', 'RUSSIA1': 'Россия 1', 'RUSSIA24': 'Россия 24'}
PERSON = {'lenin': 'Ленин', 'stalin': 'Сталин'}
MODEL_LABEL = {'M1': 'M1 · Постоянный уровень', 'M2': 'M2 · Избыточная вариация',
               'M3': 'M3 · Календарь и тренд', 'M4': 'M4 · Меняющийся уровень', 'POOL': 'Смесь четырёх моделей'}


def theme():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': GRID,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'axes.titleweight': 'normal', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'lenin-stalin-frozen-20261002',
                         'axes.axisbelow': True, 'xtick.major.size': 0, 'ytick.major.size': 0})


def clean(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=.65)
    ax.tick_params(pad=6)


def save(fig, name):
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    svg_path = FIG/f'{name}.svg'
    svg_path.write_text(svg_path.read_text(encoding='utf-8').replace(
        "font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif"), encoding='utf-8')
    fig.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def daily_figure(data, forecasts, selected, target):
    fig, axes = plt.subplots(4, 2, figsize=(13.3, 10.7))
    title = 'Упоминания по дням' if target == 'mentions' else 'Архивные записи с упоминанием'
    fig.text(.065, .985, title, fontsize=23, weight='normal', va='top')
    fig.text(.065, .95, 'История до 2 октября 2026  /  прогноз на 31 день  /  линия: среднее, полосы: 80% и 95%', color=MUTED, fontsize=10)
    last_history = pd.Timestamp('2026-08-15'); end = pd.Timestamp('2026-11-02')
    for c, channel in enumerate(m.CHANNELS):
        hist = data[(data.channel == channel) & (data.date >= last_history)].copy()
        for j, person in enumerate(m.PEOPLE):
            ax = axes[c, j]; color = COLORS[person]; key = person+'_'+target
            h = hist[key].where(hist.observed)
            ax.plot(hist.date, h, color='#A9B1B2', lw=.8, alpha=.9)
            ax.scatter(hist.date, h, color='#A9B1B2', s=4, linewidths=0)
            p = forecasts[(forecasts.model == selected) & (forecasts.scenario == 'main') & (forecasts.channel == channel) & (forecasts.target == key)].sort_values('day')
            dates = pd.to_datetime(p.day)
            ax.fill_between(dates, p.lo95, p.hi95, color=color, alpha=.09, linewidth=0)
            ax.fill_between(dates, p.lo80, p.hi80, color=color, alpha=.19, linewidth=0)
            ax.plot(dates, p['mean'], color=color, lw=1.8)
            ax.axvline(pd.Timestamp('2026-10-02 12:00'), color=MUTED, lw=.7, ls=(0, (3, 4)))
            ax.set_title(f'{LABELS[channel]}  ·  {PERSON[person]}', loc='left', fontsize=12, pad=11)
            ax.set_xlim(last_history, end); ax.set_ylim(bottom=0); clean(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=2))
            ticks = pd.to_datetime(['2026-08-15', '2026-09-01', '2026-10-03', '2026-11-02'])
            ax.set_xticks(ticks, ['15 авг', '1 сен', '3 окт', '2 ноя'])
            if c < 3:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel('в день', fontsize=9)
    fig.subplots_adjust(top=.89, bottom=.06, left=.065, right=.985, hspace=.55, wspace=.20)
    fig.text(.065, .015, 'Пробел в истории = неполный день. Полосы описывают будущие наблюдаемые счётчики, включая дневную вариацию.', fontsize=9, color=MUTED)
    save(fig, 'daily_'+target)


def month_figure(totals, selected):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    fig.text(.065, .98, 'Прогноз на месяц', fontsize=24, va='top')
    fig.text(.065, .90, '3 октября — 2 ноября 2026  /  точка: среднее  /  толстая линия: 80%  /  тонкая: 95%', color=MUTED, fontsize=10)
    for ax, target, title in zip(axes, ('mentions', 'broadcasts'), ('Все упоминания', 'Записи с ≥1 упоминанием')):
        for c, channel in enumerate(m.CHANNELS):
            for j, person in enumerate(m.PEOPLE):
                row = totals[(totals.model == selected) & (totals.scenario == 'main') & (totals.channel == channel) & (totals.target == person+'_'+target)].iloc[0]
                y = 3-c+(.12 if j == 0 else -.12); color = COLORS[person]
                ax.plot([row.lo95, row.hi95], [y, y], color=color, lw=1, alpha=.6)
                ax.plot([row.lo80, row.hi80], [y, y], color=color, lw=4, solid_capstyle='round', alpha=.4)
                ax.scatter([row['mean']], [y], s=38, color=color, zorder=4, edgecolor=PAPER, linewidth=.8)
        ax.set_title(title, loc='left', pad=15, fontsize=13)
        ax.set_yticks(range(4), [LABELS[c] for c in reversed(m.CHANNELS)])
        ax.set_ylim(-.5, 3.5); ax.set_xlim(left=0); clean(ax, 'x')
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.set_xlabel('сумма за 31 день', fontsize=9)
    fig.text(.065, .035, '● Ленин', color=COLORS['lenin'], fontsize=11)
    fig.text(.17, .035, '● Сталин', color=COLORS['stalin'], fontsize=11)
    fig.text(.31, .035, 'Интервалы сумм рассчитаны по целым совместным траекториям.', color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.73, bottom=.17, left=.09, right=.985, wspace=.34)
    save(fig, 'month_overview')


def quality_figure(test, evaluation, selected):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), gridspec_kw={'width_ratios': [1.5, 1, 1]})
    fig.text(.065, .98, 'Качество ежедневных прогнозов', fontsize=23, va='top')
    fig.text(.065, .90, '21 месяц вне периода настройки  /  январь 2025 — сентябрь 2026  /  одинаковые наблюдения для всех моделей', color=MUTED, fontsize=10)
    evaluation = evaluation.set_index('model').reindex(m.MODELS)
    by_origin = test.groupby(['origin', 'model']).scaled_crps.sum().unstack().reindex(columns=m.MODELS)
    origin_counts = test.groupby(['origin', 'model']).size().unstack().reindex(columns=m.MODELS).to_numpy()
    rng = np.random.default_rng(33)
    # Circular moving blocks of 3 calendar months preserve short serial dependence.
    blocks = rng.integers(len(by_origin), size=(2000, 7))
    indices = ((blocks[:, :, None]+np.arange(3)) % len(by_origin)).reshape(2000, 21)
    loss = by_origin.to_numpy()
    # Resample complete month blocks, preserving observation-weighted estimand
    # used by the plotted point even when month lengths/coverage differ.
    boot = loss[indices].sum(axis=1)/origin_counts[indices].sum(axis=1)
    for i, name in enumerate(m.MODELS):
        skill = 1-evaluation.loc[name, 'scaled_crps']/evaluation.loc['M1', 'scaled_crps']
        interval = np.quantile(1-boot[:, i]/boot[:, 0], [.025, .975])
        color = INK if name == selected else '#A5AFB1'
        axes[0].plot(interval, [4-i]*2, color=color, lw=2, alpha=.5)
        axes[0].scatter(skill, 4-i, s=50 if name == selected else 25, color=color, zorder=3)
    axes[0].axvline(0, color=MUTED, lw=.7, ls=(0, (3, 4)))
    axes[0].set_yticks(range(5), [MODEL_LABEL[k] for k in reversed(m.MODELS)], fontsize=9)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_title('Улучшение CRPS к M1', loc='left', fontsize=12, pad=14)
    axes[0].set_xlabel('правее = меньше ошибка', fontsize=9); clean(axes[0], 'x')
    for ax, coverage, nominal in zip(axes[1:], ('coverage80', 'coverage95'), (.8, .95)):
        for i, name in enumerate(m.MODELS):
            color = INK if name == selected else '#A5AFB1'
            ax.scatter(evaluation.loc[name, coverage], 4-i, s=50 if name == selected else 25, color=color)
        ax.axvline(nominal, color=COLORS['stalin'], lw=1, ls=(0, (3, 4)))
        ax.set_title(f'Покрытие {nominal:.0%}', loc='left', fontsize=12, pad=14)
        ax.set_yticks(range(5), [k for k in reversed(m.MODELS)])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(max(0, min(nominal, evaluation[coverage].min())-.05), 1.01)
        clean(ax, 'x'); ax.set_xlabel('доля исходов в интервале', fontsize=9)
    for ax in axes:
        ax.set_ylim(-.45, 4.5)
    fig.subplots_adjust(top=.73, bottom=.19, left=.20, right=.98, wspace=.5)
    fig.text(.065, .025, 'Полосы слева: 95% блочный bootstrap (блок 3 месяца). Выделена модель, выбранная заранее на 2024H2.', color=MUTED, fontsize=9)
    save(fig, 'backtest_quality')


def calibration_figure(test, selected):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.8))
    fig.text(.065, .98, 'Как модель описывает неопределённость', fontsize=20, va='top')
    for ax, target, title in zip(axes, ('mentions', 'broadcasts'), ('Количество упоминаний', 'Количество записей с упоминанием')):
        vals = test[(test.model == selected) & test.target.str.endswith(target)].pit
        heights, bins = np.histogram(vals, bins=np.linspace(0, 1, 11))
        ax.bar((bins[1:]+bins[:-1])/2, heights/len(vals), width=.075, color=COLORS['stalin'], alpha=.7, linewidth=0)
        ax.axhline(.1, color=INK, lw=.8, ls=(0, (3, 4)))
        ax.set_title(title, loc='left', fontsize=12); ax.set_xlim(0, 1)
        ax.set_xlabel('рандомизированный PIT', fontsize=9); ax.yaxis.set_major_formatter(PercentFormatter(1)); clean(ax)
    fig.subplots_adjust(top=.72, bottom=.22, left=.065, right=.98, wspace=.22)
    fig.text(.065, .025, 'При калибровке ожидается около 10% исходов в каждом столбце. Для дискретных счётчиков PIT рандомизирован.', color=MUTED, fontsize=9)
    save(fig, 'calibration')


def monthly_quality_figure(selected):
    table = pd.read_csv(OUT/'evaluation_monthly.csv').set_index('model')
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    fig.text(.065, .98, 'Месячные итоги требуют отдельной проверки', fontsize=21, va='top')
    fig.text(.065, .89, 'Реальная доля месячных исходов внутри интервалов  /  только полностью наблюдаемые месяцы', color=MUTED, fontsize=10)
    for ax, level in zip(axes, (80, 95)):
        nominal = level/100
        for i, name in enumerate(m.MODELS):
            ax.scatter(table.loc[name, f'coverage{level}'], 4-i,
                       s=55 if name == selected else 28, color=INK if name == selected else '#A5AFB1', zorder=3)
        ax.axvline(nominal, color=COLORS['stalin'], lw=1, ls=(0, (3, 4)))
        ax.set_title(f'Интервал {level}%', loc='left', pad=14, fontsize=12)
        ax.set_yticks(range(5), list(reversed(m.MODELS)))
        ax.set_xlim(max(0, table[f'coverage{level}'].min()-.08), 1.01); ax.set_ylim(-.45, 4.5)
        ax.xaxis.set_major_formatter(PercentFormatter(1)); clean(ax, 'x')
        ax.set_xlabel('фактическое покрытие', fontsize=9)
    fig.subplots_adjust(top=.72, bottom=.18, left=.09, right=.98, wspace=.24)
    fig.text(.065, .025, 'M3 недооценивает неопределённость месячных сумм. M4 ближе к номиналу; выбор по раннему daily CRPS сохраняется.', fontsize=9, color=MUTED)
    save(fig, 'monthly_quality')


def sensitivity_figure(totals, selected):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    fig.text(.065, .98, 'Что меняется при включении неясных контекстов', fontsize=21, va='top')
    fig.text(.065, .90, 'Средние месячные итоги и 80% интервалы  /  одна модель и те же настройки для обоих вариантов разметки', color=MUTED, fontsize=10)
    for ax, person in zip(axes, m.PEOPLE):
        for c, channel in enumerate(m.CHANNELS):
            for j, scenario in enumerate(('main', 'possible')):
                row = totals[(totals.model == selected) & (totals.scenario == scenario) & (totals.channel == channel) & (totals.target == person+'_mentions')].iloc[0]
                y = 3-c+(.10 if j == 0 else -.10); color = COLORS[person] if j == 0 else '#A5AFB1'
                ax.plot([row.lo80, row.hi80], [y, y], color=color, lw=2, alpha=.65)
                ax.scatter(row['mean'], y, color=color, s=32, zorder=3)
        ax.set_yticks(range(4), [LABELS[c] for c in reversed(m.CHANNELS)])
        ax.set_title(PERSON[person], loc='left', pad=13, fontsize=13); ax.set_ylim(-.5, 3.5); ax.set_xlim(left=0)
        ax.set_xlabel('упоминания за 31 день', fontsize=9); clean(ax, 'x')
    fig.subplots_adjust(top=.73, bottom=.19, left=.09, right=.98, wspace=.30)
    fig.text(.065, .04, 'Цвет: основной ряд. Серый: ряд с неясными ссылками и ASR. Это анализ чувствительности, не границы истинного числа.', color=MUTED, fontsize=9)
    save(fig, 'context_sensitivity')


def data_quality_figure(data):
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 5.4), sharex=True)
    fig.text(.065, .98, 'Объём текста тоже меняет наблюдаемые счётчики', fontsize=21, va='top')
    palette = ['#8A6873', '#909A64', '#B18B5F', '#347F88']
    for c, channel in enumerate(m.CHANNELS):
        d = data[(data.channel == channel) & data.observed].copy().set_index('date')
        monthly = d.resample('MS').agg({'canonical_word_count': 'sum', 'available_broadcasts': 'sum', 'stalin_mentions': 'sum'})
        monthly = monthly[d.resample('MS').size() >= 20]
        axes[0].plot(monthly.index, monthly.canonical_word_count/monthly.available_broadcasts, color=palette[c], lw=1.5, label=LABELS[channel])
        axes[1].plot(monthly.index, 100000*monthly.stalin_mentions/monthly.canonical_word_count, color=palette[c], lw=1.5)
    axes[0].set_ylabel('слов на запись', fontsize=9); axes[1].set_ylabel('Сталин / 100 тыс. слов', fontsize=9)
    axes[0].legend(frameon=False, ncols=4, loc='upper left', bbox_to_anchor=(0, 1.35), fontsize=9)
    for ax in axes:
        clean(ax); ax.set_ylim(bottom=0)
    axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.subplots_adjust(top=.70, bottom=.16, left=.10, right=.98, hspace=.22)
    fig.text(.065, .025, 'Месяцы с ≥20 полными днями. Нормирование по словам не устраняет ошибки транскрибации.', color=MUTED, fontsize=9)
    save(fig, 'text_volume')


def posterior_checks(data, manifest):
    from run_joint_forecast import fit_cache, BASE_SEED
    seed = BASE_SEED+99999
    f2 = fit_cache(data, False, seed, manifest['source_sha256'], suffix='main')
    f3 = fit_cache(data, True, seed+1000, manifest['source_sha256'], suffix='main')
    dates = pd.date_range('2025-10-03', '2026-10-02')
    obs = data[data.date.isin(dates)]; rows = []
    # In-sample posterior predictive adequacy, deliberately separate from test scores.
    forecasts = {'M1': m.conjugate_predict(data, dates, 1500, 34),
                 'M2': m.static_predict(f2, dates, 1500, 35), 'M3': m.static_predict(f3, dates, 1500, 36)}
    functions = {'mean': lambda a, axis: np.mean(a, axis=axis),
                 'variance': lambda a, axis: np.var(a, axis=axis),
                 'zero_share': lambda a, axis: np.mean(a == 0, axis=axis),
                 'p95': lambda a, axis: np.quantile(a, .95, axis=axis)}
    for model, paths in forecasts.items():
        for c, channel in enumerate(m.CHANNELS):
            d = obs[obs.channel == channel].sort_values('date'); ok = d.observed.to_numpy()
            for person in m.PEOPLE:
                for target in ('mentions', 'broadcasts'):
                    key = person+'_'+target; y = d.loc[ok, key].to_numpy(); s = paths[key][:, ok, c]
                    for stat, fn in functions.items():
                        rep = fn(s, 1); actual = float(fn(y, 0)); lo, hi = np.quantile(rep, [.025, .975])
                        rows.append(dict(model=model, channel=channel, target=key, statistic=stat, observed=actual,
                                         predictive_median=float(np.median(rep)), lo95=float(lo), hi95=float(hi),
                                         outside95=bool(actual < lo or actual > hi)))
    pd.DataFrame(rows).to_csv(OUT/'posterior_predictive_checks.csv', index=False)
    # M3 calendar effects are conditional coefficients, not causal estimates.
    calendar_rows = []
    for person in m.PEOPLE:
        fit = f3['B_'+person]
        for name in ('exact_nov7', 'exact_may8', 'exact_may9'):
            values = np.exp(fit['theta'][:, fit['names'].index(name)])
            q = np.quantile(values, [.025, .1, .5, .9, .975])
            calendar_rows.append(dict(person=person, effect=name, lo95=q[0], lo80=q[1], median=q[2], hi80=q[3], hi95=q[4]))
    pd.DataFrame(calendar_rows).to_csv(OUT/'calendar_effects.csv', index=False)


def monthly_evaluation():
    d = pd.read_csv(OUT/'backtest_monthly.csv')
    for level in (80, 95):
        d[f'coverage{level}'] = (d[f'lo{level}'] <= d.actual) & (d.actual <= d[f'hi{level}'])
        d[f'width{level}'] = d[f'hi{level}']-d[f'lo{level}']
    def metrics(g):
        return pd.Series(dict(complete_month_targets=len(g), origins=g.origin.nunique(),
                              rmse=float(np.sqrt(g.se.mean())), crps=g.crps.mean(),
                              coverage80=g.coverage80.mean(), coverage95=g.coverage95.mean(),
                              width80=g.width80.mean(), width95=g.width95.mean()))
    d.groupby('model').apply(metrics, include_groups=False).reset_index().to_csv(OUT/'evaluation_monthly.csv', index=False)
    d.groupby(['model', 'channel', 'target']).apply(metrics, include_groups=False).reset_index().to_csv(OUT/'evaluation_monthly_by_series.csv', index=False)


def inline_svg(name):
    text = (FIG/f'{name}.svg').read_text(encoding='utf-8')
    text = text[text.index('<svg'):]
    return re.sub(r'<metadata>.*?</metadata>', '', text, flags=re.S)


def findings(manifest, evaluation, totals):
    selected = manifest['selected_model']; ev = evaluation.set_index('model')
    skill = 1-ev.loc[selected, 'scaled_crps']/ev.loc['M1', 'scaled_crps']
    return dict(selected=selected, selected_label=MODEL_LABEL[selected], skill=skill,
                coverage80=ev.loc[selected, 'coverage80'], coverage95=ev.loc[selected, 'coverage95'])


def html_report(manifest, evaluation, totals):
    f = findings(manifest, evaluation, totals); selected = f['selected']
    table = []
    for channel in m.CHANNELS:
        for person in m.PEOPLE:
            vals = []
            for target in ('mentions', 'broadcasts'):
                row = totals[(totals.scenario == 'main') & (totals.model == selected) & (totals.channel == channel) & (totals.target == person+'_'+target)].iloc[0]
                vals.append(f"<strong>{row['mean']:.0f}</strong><small>{row.lo80:.0f}–{row.hi80:.0f}</small>")
            table.append(f'<tr><td>{LABELS[channel]}</td><td><span style="color:{COLORS[person]}">{PERSON[person]}</span></td><td>{vals[0]}</td><td>{vals[1]}</td></tr>')
    def panel(name, caption):
        return f'<figure>{inline_svg(name)}<figcaption>{caption}</figcaption></figure>'
    text = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ленин и Сталин в телеэфире · прогноз</title><style>
:root{{--paper:#FCFBF8;--ink:#243C45;--muted:#748188;--line:#E4E6E2;--teal:#277978}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,'Segoe UI',sans-serif}}main{{max-width:1200px;margin:auto;padding:55px 34px 90px}}header{{max-width:1000px}}.eyebrow{{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}}h1{{font:normal 52px/1.12 Georgia,serif;letter-spacing:-1.6px;margin:20px 0}}.lead{{max-width:820px;font-size:19px;color:#52666D}}nav{{display:flex;gap:24px;border-bottom:1px solid var(--line);padding:16px 0;margin:28px 0}}a{{color:var(--teal);text-decoration:none}}a:hover{{text-decoration:underline}}nav a{{color:var(--ink);font-size:13px}}h2{{font:normal 32px/1.2 Georgia,serif;margin:55px 0 18px}}h3{{font-size:18px;font-weight:500}}.cards{{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:24px;margin:30px 0}}.card{{border-top:2px solid var(--line);padding-top:14px}}.card small{{color:var(--muted);display:block;font-size:12px}}.value{{font-size:26px;line-height:1.4;margin:7px 0}}.note{{border-left:2px solid #B46A48;padding:3px 0 3px 18px;color:#5C6A6F;font-size:14px;max-width:950px}}figure{{margin:30px 0 40px}}figure svg{{display:block;width:100%;height:auto}}figcaption{{color:var(--muted);font-size:12px;max-width:950px;margin-top:8px}}.tabs{{display:flex;gap:8px;margin:20px 0}}button{{border:1px solid var(--line);padding:8px 18px;background:transparent;border-radius:24px;cursor:pointer;color:var(--ink);font:inherit;font-size:13px}}button.active{{background:var(--ink);color:var(--paper);border-color:var(--ink)}}.tab[hidden]{{display:none}}table{{width:100%;border-collapse:collapse;font-size:14px}}th{{font-weight:400;color:var(--muted);text-align:left;font-size:12px}}td,th{{border-bottom:1px solid var(--line);padding:12px 15px 12px 0}}td small{{display:inline;margin-left:14px;color:var(--muted);font-size:12px}}td strong{{font-weight:500}}.methods{{display:grid;grid-template-columns:1fr 1fr;gap:12px 36px}}.methods p{{font-size:14px;color:#52666D}}.formula{{font:22px/1.6 Georgia,serif;padding:22px 0;border-block:1px solid var(--line);margin:24px 0}}footer{{margin-top:50px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px}}code{{font-size:.9em}}@media(max-width:750px){{main{{padding:30px 15px}}h1{{font-size:36px}}.cards,.methods{{grid-template-columns:1fr}}nav{{gap:12px;flex-wrap:wrap}}.value{{font-size:23px}}td small{{display:block;margin:0}}}}
</style><main><header><div class="eyebrow">Купча Софья · Пальцева Ксения / Forecasting project</div><h1>Ленин и Сталин<br>в телеэфире</h1><p class="lead">Прогноз числа упоминаний и архивных записей с упоминанием на четырёх российских телеканалах. 3 октября — 2 ноября 2026 года.</p><p class="note">Информационная граница: 2 октября 2026. Это воспроизведение прогноза из зафиксированной точки прошлого: исходы после неё не использовались. Объект измерения — предварительно отобранные контексты фамилий в архивных транскриптах.</p></header>
<nav><a href="#forecast">Прогноз</a><a href="#quality">Качество</a><a href="#sensitivity">Чувствительность</a><a href="#methods">Как это работает</a></nav>
<div class="cards"><div class="card"><small>Выбрана на ранних данных, до теста</small><div class="value">{html.escape(f['selected_label'])}</div><small>Период настройки: июль — декабрь 2024</small></div><div class="card"><small>CRPS относительно базовой модели на тесте</small><div class="value">{f['skill']:+.1%}</div><small>Положительное значение означает улучшение</small></div><div class="card"><small>Фактическое покрытие интервалов 80% / 95%</small><div class="value">{f['coverage80']:.1%} / {f['coverage95']:.1%}</div><small>Проверка на 21 историческом месяце</small></div></div>
<section id="forecast"><h2>Месячный итог и ежедневная траектория</h2><p>Средний прогноз отвечает на вопрос о типичном объёме. Полосы показывают, насколько широко могут разойтись будущие наблюдаемые значения.</p>
{panel('month_overview', 'Месячный интервал рассчитан после суммирования каждой траектории. Складывать границы ежедневных интервалов было бы неверно.')}
<table><thead><tr><th>Канал</th><th>Фигура</th><th>Упоминания · среднее и 80%</th><th>Записи с упоминанием · среднее и 80%</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<div class="tabs" role="group" aria-label="Выбор прогнозируемой величины"><button class="active" onclick="showTab('mentions',this)">Все упоминания</button><button onclick="showTab('broadcasts',this)">Записи с упоминанием</button></div>
<div id="mentions" class="tab">{panel('daily_mentions', 'Цвета фигур одинаковы во всех графиках: терракотовый — Ленин, зелёный — Сталин. Серым показаны реальные счётчики до границы прогноза.')}</div>
<div id="broadcasts" class="tab" hidden>{panel('daily_broadcasts', 'Счётчик записей с упоминанием не превышает число записей в той же сгенерированной траектории.')}</div></section>
<section id="quality"><h2>Усложнение должно оправдываться прогнозом</h2><p>Модели обучались заново на прошлом перед каждым историческим месяцем. Настройки и выбор модели зафиксированы на отдельном раннем периоде; результаты 2025–2026 годов служат внешней проверкой.</p>
{panel('backtest_quality', 'CRPS оценивает всё распределение; RMSE, покрытие и ширина интервалов сохранены в таблицах. Межсерийный показатель делит CRPS на max(1, среднее ряда в обучении), чтобы Россия 24 не определяла весь результат своим масштабом.')}
{panel('calibration', 'PIT проверяет агрегированную калибровку, а не независимость ошибок. Для дискретных счётчиков включающие целые границы интервалы могут иметь покрытие выше номинала. Таблицы дают отдельные результаты для всех 16 целей.')}</section>
<section>{panel('monthly_quality', 'Высокое дневное покрытие не переносится автоматически на суммы. Серийная зависимость особенно важна для месячных итогов; M3 показывает недостаточное покрытие, M4 — более широкие интервалы и покрытие ближе к номиналу.')}</section>
<section><h3>Что следует из проверки</h3>{(OUT/'findings.html').read_text(encoding='utf-8')}</section>
<section id="sensitivity"><h2>Что зависит от способа измерения</h2><p>Сомнительные ссылки и ошибки распознавания речи могут менять ряд ещё до моделирования. Расширенная версия включает неясные контексты и обучается отдельно с теми же настройками.</p>
{panel('context_sensitivity', 'Этот диапазон не покрывает все ошибки классификации и ASR. Ни основной, ни расширенный ряд не является доказанной границей истинного количества.')}
{panel('text_volume', 'Количество архивных записей и длина их транскриптов — разные характеристики наблюдаемого объёма. Изменение счётчиков нельзя автоматически трактовать как изменение политической позиции.')}</section>
<section id="methods"><h2>Согласованный прогноз</h2><div class="formula">Число записей N → записи с упоминанием B | N → упоминания C = B + E | B</div><p>Для каждой траектории: 0 ≤ B ≤ N и C ≥ B. При B = 0 упоминаний тоже нет. Число записей N общее для обоих людей; будущий знаменатель прогнозируется, а не берётся из будущих данных.</p><div class="methods">
<div><h3>M1 · Сопряжённая базовая модель</h3><p>Постоянный Poisson-уровень числа записей; Binomial-доля записей с упоминанием; Poisson-добавка внутри положительных записей. Gamma- и Beta-априорные распределения обновляются аналитически.</p></div>
<div><h3>M2 · Вариация между днями</h3><p>Negative binomial для числа записей и повторных упоминаний; Beta-binomial для доли упоминающих записей. Полный статический posterior рассчитывается Metropolis–Hastings.</p></div>
<div><h3>M3 · Календарная иерархия</h3><p>Общий уровень, сжатые отклонения каналов, тренд, месяцы и дни недели, точные даты и соседние дни. Масштабы Normal-приоров заданы заранее; это частичный pooling с фиксированными гиперпараметрами.</p></div>
<div><h3>M4 · Меняющийся уровень</h3><p>К M3 добавлены случайные блуждания на log/logit-шкалах. Фильтр распространяет частицы, обновляет веса и ресэмплирует при ESS &lt; половины числа частиц. Это явно модульное приближение, а не полный совместный Bayes.</p></div>
<div><h3>Смесь · Неопределённость между моделями</h3><p>Равновесная смесь выбирает целую траекторию из одной модели. Средние и квантили не усредняются отдельно; согласованность счётчиков сохраняется.</p></div>
<div><h3>Проверка вычислений</h3><p>Четыре цепи, rank-normalized split R-hat &lt; 1,01, bulk/tail ESS ≥ 400. Отдельно проверены функции вероятности, градиенты, CRPS, хронология и все совместные траектории.</p></div></div>
<details><summary>Ограничения и воспроизводимость</summary><p>День определяется началом архивной записи в MSK. Учитываются все программы, реклама и повторы в другое время. Пропуски не заменены нулями. Это прогноз архивных счётчиков при сопоставимом измерении; он не предсказывает политический курс или причинный эффект событий. Семантическая точность правил не измерена независимой человеческой разметкой. M4 использует 365-дневное окно состояний и cut posterior. Между Лениным и Сталиным остаётся немоделируемая зависимость сверх общего N. DM-тесты с 21 блоком приблизительны; результаты скорректированы Holm. Месячные исходы оцениваются только при полном наблюдении месяца.</p><p>Полный расчёт: <code>python analysis/run_joint_forecast.py</code>. Отчёт для сдачи: <a href="final_project.ipynb">final_project.ipynb</a>. <a href="forecast_daily.csv">Ежедневные прогнозы</a> · <a href="forecast_month.csv">Месячные итоги</a> · <a href="evaluation_by_series.csv">Оценка по всем целям</a> · <a href="README.md">Методика и файлы</a>.</p></details></section>
<footer>Данные: сохранённые транскрипты GDELT и исправленный локальный набор. Последнее наблюдение: 02.10.2026. Seed: {manifest['seed']}. Все графики доступны отдельно в SVG и PNG. Расчёт выполнен локально, без внешнего API.</footer></main>
<script>function showTab(id,button){{document.querySelectorAll('.tab').forEach(e=>e.hidden=e.id!==id);document.querySelectorAll('.tabs button').forEach(e=>e.classList.remove('active'));button.classList.add('active')}}</script></html>'''
    (OUT/'forecast_report.html').write_text(text, encoding='utf-8')


def build_notebook(manifest):
    import nbformat as nb
    n = nb.v4.new_notebook(); cells = []
    def md(s): cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s): cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
    md(r'''
    # Forecasting daily mentions of Lenin and Stalin on Russian television
    **Купча Софья · Пальцева Ксения**

    Воспроизводимый итоговый проект. Цели: количество упоминаний фамилии C и количество архивных записей с ≥1 упоминанием B; два человека, четыре канала, 16 рядов. Прогноз: **3 октября — 2 ноября 2026**. Информация ограничена **2 октября 2026**: это frozen-cutoff прогноз, а не заявление о сегодняшнем информационном наборе.

    Инструкция преподавателя требует `.ipynb`, объяснения каждого вычислительного шага и доступных данных. Все модели и процедуры находятся в поставляемых `analysis/joint_count_models.py`, `run_joint_forecast.py`, `verify_joint_forecast.py`, `forecast_artifacts.py`. Этот ноутбук вызывает те же функции; кэш можно полностью перестроить. Исходная заявка: `output/proposal/forecasting_proposal.tex`.
    ''')
    md('''
    ## 1. Запуск и данные
    Устанавливаем `requirements-forecast.txt` из корня проекта. Данные читаются из `data/cleaned/tv_daily_targets.csv.gz`; этот сжатый набор включён в проект. Восстановление из оригиналов: `python analysis/prepare_clean_data.py`. Правила отбора и ограничения описаны в `output/data_cleaning/README.md`; нейросетевой эксперимент отменён и на эти результаты не влияет.
    ''')
    code('''
    from pathlib import Path
    import sys, json
    import numpy as np
    import pandas as pd
    from IPython.display import display, SVG, HTML
    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next(p for p in candidates if (p/'analysis/joint_count_models.py').exists())
    sys.path.insert(0, str(ROOT/'analysis'))
    import joint_count_models as models
    from run_joint_forecast import run_experiment
    from verify_joint_forecast import verify
    OUT = ROOT/'output/joint_forecast'
    data, input_sha = models.load_data()
    display(data.groupby('channel').agg(days=('day','size'), observed=('observed','sum'),
                                      mean_records=('available_broadcasts','mean')))
    print('Last observation:', data.date.max().date(), '| input SHA-256:', input_sha)
    ''')
    md(r'''
    Полнота дня проверяется по UTC-инвентарям и доступным транскриптам; агрегация использует MSK-день начала записи. Неполные дни отсутствуют в likelihood и оценке, они не становятся нулями. Удаляются лишь доказанные совпадения упоминаний в перекрывающихся записях. Основная метка `person_likely` предварительная; расширенный ряд добавляет неясные ссылки и ASR.

    Политическая риторика мотивирует задачу, но прогнозируемый объект уже: счётчики фамилий в архивных текстах. Длина транскриптов, повторы и ошибки ASR ограничивают такую интерпретацию.
    ''')
    code("if (OUT/'figures/text_volume.svg').exists():\n    display(SVG(filename=str(OUT/'figures/text_volume.svg')))\nelse:\n    print('График объёма текста будет создан после расчёта в разделе 4.')")
    md(r'''
    ## 2. Генеративная зависимость счётчиков
    Комментарий преподавателя требует уважать зависимость N, B и C. Используем
    $$p(N,B_L,C_L,B_S,C_S)=p(N)\prod_{r\in\{L,S\}}p(B_r\mid N)\,p(C_r\mid B_r).$$
    У каждого положительного архива есть хотя бы одно упоминание: $C=B+E$. При $B=0$ полагаем $E=0$. Поэтому $0\le B\le N$ и $C\ge B$ во всех реализациях. Это не независимые прогнозы, исправленные после расчёта. Остаточная связь между двумя людьми сверх общего N пока не моделируется.

    ### M1: аналитический Bayesian baseline (L2)
    $N_t\sim Poisson(\lambda_N)$, $B_t\mid N_t\sim Binomial(N_t,p)$,
    $E_t\mid B_t\sim Poisson(B_t\lambda_E)$.
    Независимые априорные распределения по каналам: $\lambda_N\sim Gamma(2,0.1)$, $p\sim Beta(1,19)$, $\lambda_E\sim Gamma(1,1)$; второй аргумент Gamma — rate.
    Posterior: $Gamma(2+\sum N,0.1+n)$; $Beta(1+\sum B,19+\sum(N-B))$; $Gamma(1+\sum E,1+\sum B)$. Одна posterior-параметризация сохраняется на всю месячную траекторию, что переносит неопределённость параметров в месячный итог.

    ### M2: избыточная вариация (L2–L3)
    $N_t\sim NB(\mu_{Nt},k_N)$, $B_t\mid N_t\sim BetaBinomial(N_t,p_t\phi,(1-p_t)\phi)$.
    $E_t\mid B_t>0\sim NB(B_t\lambda_{Et},B_t k_E)$. Последняя формула есть сумма B независимых дополнительных счётчиков NB с одинаковым per-record размером k. В параметризации NB $Var(Y)=\mu+\mu^2/k$.
    Для каждого компонента используются общий уровень и ортонормальные контрасты каналов. Dispersion разделяется каналами внутри компонента. Различия людей имеют отдельные модели B и E.
    Приоры общего уровня (SD=1.5): N — Normal(log 20,1.5²), B — Normal(logit 0.05,1.5²), E — Normal(0,1.5²). Приоры log-dispersion с тем же SD: log k_N — Normal(log 5,1.5²), log phi — Normal(log 20,1.5²), log k_E — Normal(0,1.5²). Время центрируется по обучающим датам, не по будущим исходам.

    ### M3: календарь, тренд и partial pooling (L4)
    В log/logit-предиктор добавляются линейный тренд в годах, 11 суммирующихся в ноль месячных и 6 недельных контрастов. Календарь общий для каналов внутри компонента; отклонения каналов сжимаются к общему уровню. Контрасты месяца имеют $N(0,0.35^2)$, дня недели $N(0,0.25^2)$, канала $N(0,0.8^2)$; тренд $N(0,0.35^2)$.
    Точные даты и дни на расстоянии 1–2 дней моделируются отдельно: 7 ноября, 8 мая, 9 мая, рождение и смерть Ленина, смерть и официальная дата рождения Сталина. Коэффициенты точной даты $N(0,0.7^2)$, соседних дней $N(0,0.5^2)$.
    Это prior-specified partial pooling: гипермасштабы фиксированы заранее, а не оценены из данных. Близкие праздники могут быть слабо различимы на трёх годовых повторах. 7 ноября вне финального горизонта; его индикатор присутствует в модели, а не добавлен в прогноз искусственно.
    ''')
    md(r'''
    ## 3. Bayesian computation и последовательное обновление
    Статические M2/M3 используют Metropolis–Hastings с точным likelihood × prior (L3). Мода и численный Hessian задают многомерное Student-t предложение с df=12; это не замена posterior на Gaussian. Начальные точки и случайные потоки четырёх цепей различаются. Warmup отбрасывается. При $\hat R\ge1.01$ или bulk/tail ESS <400 расчёт повторяется с удвоенным числом итераций; неудачная сходимость останавливает workflow.

    **M4 (L5):** $s_{t}=s_{t-1}+\epsilon_t$, $\epsilon_t\sim N(0,q^2)$ на log/logit-шкале каждого компонента/канала. Статические параметры берутся из M3. Для каждого такого draw работает отдельный условный particle filter: propagate, likelihood weighting, normalize, systematic resampling при ESS < P/2. В неполный день выполняется только propagate; при B=0 параметр избыточных упоминаний не обновляется. Начало 365-дневного окна: $s\sim N(0,0.25^2)$.
    Это **cut/modular approximation**: static posterior уже оценён на том же обучении, затем состояния фильтруются условно, без переоценки весов static draws. Уверенность может отличаться от полного совместного Bayes. Динамический уровень распространяется и на будущие дни, поэтому интервалы должны учитывать будущие инновации.

    **Смесь (L6):** с вероятностью 1/4 выбираем целую траекторию M1, M2, M3 или M4. Это linear pool распределений, не усреднение квантилей и не BMA. Равные веса не подгоняются на тесте.
    ''')
    md('''
    ## 4. Честная временная проверка
    Шесть validation-месяцев: июль–декабрь 2024. На них выбираем q ∈ {0.015,0.04,0.08} по среднему месячных средних scaled CRPS модели M4 (равный вес месяцев), затем выбираем один из M1–M4/смеси по всем дневным forecast cases. Внешний test: январь 2025–сентябрь 2026, 21 непересекающийся календарный месяц. Перед каждым origin переобучаемся на расширяющемся прошлом. Случайного train/test split нет.

    Основная оценка: ежедневный CRPS; для сопоставления масштабов делим его на max(1, среднее соответствующего ряда в обучении). Отдельно сохраняем обычный CRPS, RMSE по posterior predictive mean, покрытие и ширину интервалов 80/95%, PIT, оценки по горизонту. Для месячных исходов требуем полного наблюдения всех дней.

    Следующая ячейка может пересчитать ВСЕ модели, backtest и чувствительность. По умолчанию используются результаты завершённого расчёта; `RUN_REBUILD=True` принудительно переоценивает posterior. Эквивалент из терминала: `python analysis/run_joint_forecast.py --rebuild`.
    ''')
    code('''
    RUN_REBUILD = False
    if RUN_REBUILD or not (OUT/'manifest.json').exists():
        run_experiment(jobs=1, rebuild=RUN_REBUILD)
        from forecast_artifacts import build_artifacts
        build_artifacts()
    verification = verify()
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    display(pd.Series(selection, name='Frozen validation selection'))
    print('Verification:', verification['status'], '| max Rhat:', verification['max_rhat'])
    ''')
    md(r'''
    CRPS вычисляется как $E|X-y|-\frac12E|X-X'|$ с точной формулой для эмпирического predictive-распределения. Число реализаций в backtest — 2048, финально — 30000. Малые различия scores могут отражать Monte Carlo noise. Неопределённость skill на рисунке получена circular moving-block bootstrap по месяцам, длина блока 3; все цели внутри месяца остаются вместе.
    Рандомизированный PIT использует $F(y^-)+U\,[F(y)-F(y^-)]$ для дискретного распределения. Плоская объединённая гистограмма не устанавливает ни условную калибровку, ни независимость ошибок (S2). Здесь прогноз многодневный; исходы внутри месячной траектории связаны. Из-за целочисленных границ интервалы счётчиков могут иметь покрытие выше номинала даже у правильной модели; поэтому одновременно смотрим CRPS, PIT и ширину.
    ''')
    code('''
    evaluation = pd.read_csv(OUT/'evaluation.csv')
    display(evaluation[['model','rmse','crps','scaled_crps','coverage80','coverage95','width80','width95']].round(4))
    display(SVG(filename=str(OUT/'figures/backtest_quality.svg')))
    display(SVG(filename=str(OUT/'figures/calibration.svg')))
    ''')
    md('''
    Diebold–Mariano сравнивает средние парные потери по 21 месячному блоку. Используем Bartlett HAC с lag=1 для зависимости между соседними месяцами. Наблюдения/каналы/люди внутри месяца не объявляются независимыми. Положительная разность означает выигрыш относительно M1. Поправка Holm применяется отдельно к четырём сравнениям по каждому из двух типов потерь. При 21 блоке p-values приблизительны; это описательная проверка различий, а не доказательство универсального превосходства.
    ''')
    code('''
    display(pd.read_csv(OUT/'dm_tests.csv').round(4))
    display(pd.read_csv(OUT/'evaluation_by_horizon.csv').round(4))
    display(pd.read_csv(OUT/'evaluation_monthly.csv').round(3))
    display(SVG(filename=str(OUT/'figures/monthly_quality.svg')))
    ''')
    md('''
    ## 5. Итоговый прогноз
    После фиксации выбора переобучаемся на всех полных днях до 2 октября 2026. Выдаём все модели и заранее выбранный основной прогноз. Месячные интервалы получаем **по суммам каждой совместной траектории**, а не сложением квантилей по дням. Показываем predictive intervals для наблюдаемых счётчиков, а не только uncertainty среднего уровня.
    ''')
    code('''
    monthly = pd.read_csv(OUT/'forecast_month.csv')
    main = monthly[(monthly.scenario=='main') & (monthly.model==manifest['selected_model']) & (monthly.target!='N')]
    display(main[['channel','target','mean','lo80','hi80','lo95','hi95']].round(1))
    display(SVG(filename=str(OUT/'figures/month_overview.svg')))
    display(SVG(filename=str(OUT/'figures/daily_mentions.svg')))
    display(SVG(filename=str(OUT/'figures/daily_broadcasts.svg')))
    ''')
    md('''
    ## 6. Проверки адекватности и чувствительности
    Posterior predictive checks сравнивают реальные mean/variance/zero-share/p95 за последний год с репликациями M1–M3. Это in-sample проверка модели, **не внешний test**. Неудачные проверки сохраняются и обсуждаются; правильная сходимость MCMC не означает правильной модели. Отдельно смотрим чувствительность particle filter к большему числу static draws и частиц; разница включает фильтровый Monte Carlo error, который нельзя оценить только по SE финальных траекторий.

    Расширенный ряд неясных контекстов переоценивается с теми же моделями и q, без нового выбора на тесте. Это сценарий измерения, не доказанная верхняя граница. Интервалы основного прогноза не интегрируют всю ошибку классификации.
    Исторический backtest ретроспективный: используется одна исправленная версия архива и единое определение меток. Исторические real-time vintages транскриптов и меток отсутствуют; их доступность именно в момент каждого origin не подтверждена.
    ''')
    code('''
    checks = pd.read_csv(OUT/'posterior_predictive_checks.csv')
    display(checks.groupby(['model','statistic']).outside95.agg(['sum','count']))
    display(pd.read_csv(OUT/'particle_sensitivity.csv').round(3))
    display(SVG(filename=str(OUT/'figures/context_sensitivity.svg')))
    display(pd.read_csv(OUT/'calendar_effects.csv').round(2))
    ''')
    md('''
    Коэффициенты календаря в последней таблице — odds ratios точного дня в B-компоненте M3, условно на остальные признаки; это не причинные эффекты и не мультипликаторы общего числа упоминаний.

    ## 7. Выводы и ограничения
    ''')
    code("display(HTML((OUT/'findings.html').read_text(encoding='utf-8')))")
    md('''
    ## 8. Связь с курсом и воспроизводимость
    | Материал | Применение в проекте |
    |---|---|
    | L1 / S1 Forecast evaluation | Среднее для squared loss; CRPS, calibration/sharpness, одинаковые forecast cases |
    | L2 / S2 Bayesian forecasting | Сопряжённый baseline, posterior predictive, PPC; uncertainty параметров и наблюдений |
    | L3 / S3 Bayesian computation | Точный MH, несколько цепей, warmup, Rhat и ESS, независимые численные проверки |
    | L4 / S4 Hierarchical models | Общий уровень и shrinkage каналов/календаря, явные фиксированные гипермасштабы |
    | L5 / S5 Sequential forecasting | Латентный уровень, propagate–weight–resample, ESS и пропущенные измерения |
    | L6 / S6 Forecast aggregation | Смесь распределений, целые траектории, отдельный внешний test |
    | Комментарий Марата | Праздничные индикаторы; разграничение человека и названного объекта; N→B→C |

    Техническая сверка параметризаций: [SciPy NB](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html), [SciPy Beta-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html), [Stan: диагностика цепей](https://mc-stan.org/learn-stan/diagnostics-warnings.html). Основные материалы — предоставленные лекции и семинары курса.

    Все исходные вычисления доступны в четырёх Python-модулях выше. Их SHA-256 и версии окружения записаны в manifest; данные для модели доступны локально в исправленной сжатой таблице. Фиксированные seed обеспечивают повторяемые случайные потоки. Корпус оригиналов можно перепроверить отдельно по документации очистки; для запуска прогноза Google Cloud и внешние API не требуются.
    ''')
    code('''
    display(pd.Series(manifest['versions'], name='Environment'))
    display(pd.Series(manifest['code_sha256'], name='Executable SHA-256'))
    print('Data SHA-256:', manifest['source_sha256'])
    ''')
    n.cells = cells; n.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                 'language_info': {'name': 'python', 'version': manifest['versions']['python']}}
    nb.write(n, OUT/'final_project.ipynb')


def write_findings(manifest, evaluation, totals):
    f = findings(manifest, evaluation, totals); selected = f['selected']
    checks = pd.read_csv(OUT/'posterior_predictive_checks.csv')
    nfail = int(checks[(checks.model == 'M3') & checks.outside95].shape[0])
    sensitivity = pd.read_csv(OUT/'particle_sensitivity.csv')
    max_change = sensitivity.relative_difference.abs().max()
    monthly = pd.read_csv(OUT/'evaluation_monthly.csv').set_index('model')
    text = f'''<p>На раннем периоде выбрана <b>{html.escape(f['selected_label'])}</b>. На внешних 21 месяцах её scaled CRPS относительно M1 изменился на <b>{f['skill']:+.1%}</b> в шкале улучшения. Покрытие 80% и 95% интервалов составило <b>{f['coverage80']:.1%}</b> и <b>{f['coverage95']:.1%}</b>. Выбор не изменялся после просмотра внешнего теста.</p><p>В posterior predictive checks M3 вне 95% интервала репликаций оказались {nfail} из 64 проверяемых сводок. Это указывает, какие свойства данных модель воспроизводит недостаточно; качество вероятностного прогноза следует оценивать вместе с этими несоответствиями. При увеличении числа частиц максимальная относительная разница средних месячных итогов M4 составила {max_change:.1%}; финальный Monte Carlo error включает неопределённость фильтров, а не только траекторий.</p><p>Прогноз относится к архивным счётчикам при сопоставимом измерении. Он не даёт причинного вывода о политическом курсе. Семантическая точность правил, ошибки ASR, длина транскриптов, зависимость между двумя людьми и модульное приближение M4 остаются ограничениями. Основные и расширенные контексты показаны раздельно.</p>'''
    text += f'''<p><b>Для месячных сумм интервалы основной M3 слишком узкие:</b> фактическое покрытие составило {monthly.loc[selected, 'coverage80']:.1%} и {monthly.loc[selected, 'coverage95']:.1%} при номинальных 80% и 95%. У M4 оно ближе к номиналу: {monthly.loc['M4', 'coverage80']:.1%} и {monthly.loc['M4', 'coverage95']:.1%}. Ежедневная точность M3/M4 практически одинакова, но динамика лучше описывает часть неопределённости месячных сумм. Основной выбор M3 сохраняется по заранее заданному daily CRPS; интервалы не перекалибровываются на внешнем тесте. Это ограничение нужно учитывать при использовании месячных итогов.</p>'''
    (OUT/'findings.html').write_text(text, encoding='utf-8')


def build_artifacts():
    FIG.mkdir(parents=True, exist_ok=True); theme()
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    forecasts = pd.read_csv(OUT/'forecast_daily.csv'); totals = pd.read_csv(OUT/'forecast_month.csv')
    test = pd.read_csv(OUT/'backtest_daily.csv.gz'); evaluation = pd.read_csv(OUT/'evaluation.csv')
    data, _ = m.load_data(); selected = manifest['selected_model']
    daily_figure(data, forecasts, selected, 'mentions'); daily_figure(data, forecasts, selected, 'broadcasts')
    month_figure(totals, selected); quality_figure(test, evaluation, selected)
    calibration_figure(test, selected); sensitivity_figure(totals, selected); data_quality_figure(data)
    monthly_evaluation(); monthly_quality_figure(selected)
    posterior_checks(data, manifest); write_findings(manifest, evaluation, totals)
    html_report(manifest, evaluation, totals); build_notebook(manifest)
    print('Styled SVG/PNG figures, standalone HTML and reproducible notebook created.', flush=True)


if __name__ == '__main__':
    build_artifacts()
