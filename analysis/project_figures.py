"""Figures of the final model for the report and the presentation.

python analysis/project_figures.py

The final model is M3HR: the calendar-and-trend model M3 with an estimated
shrinkage scale of month-of-year effects and a month level shared by channels
and days (L4, pp. 30-31; HMC, L3). Its structure was chosen after the outer
test, so test scores of the final model are optimistic (L6, p. 35).
Reads the cached test trajectories and scores of run_forecast_v2.py, the final
trajectories and the cached final posterior; writes SVG and PNG to output/figures.
Monthly intervals are quantiles of trajectory sums (L1), never sums of quantiles.
With SLIDES set (build_presentation2.py) the figures go there as PNG without
their own title, because the slide carries it. LANG = 'en' switches the labels
to English (build_presentation2_en.py).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.linalg import helmert

import evaluate_forecast_v2 as e
import forecast_artifacts as fa
import joint_count_models as m
import run_forecast_v2 as rv
import run_joint_forecast as rj

FIG = m.ROOT/'output/figures'
SLIDES = None  # directory for headerless slide PNGs instead of FIG
LANG = 'ru'  # 'en' for the English deck
FINAL = 'M3HR'
INK, MUTED, PAPER, GRID = fa.INK, fa.MUTED, fa.PAPER, fa.GRID
LIGHT = '#A5AFB1'
MONTHS = (['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'],
          ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
WEEKDAYS = (['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
LADDER = ['M1', 'M2', 'M3', 'M4', FINAL]
LADDER_LABEL = {'M1': ('M1 · постоянный уровень', 'M1 · constant level'), 'M2': ('M2 · сверхдисперсия', 'M2 · overdispersion'),
                'M3': ('M3 · календарь и тренд', 'M3 · calendar and trend'), 'M4': ('M4 · меняющийся уровень', 'M4 · time-varying level'),
                FINAL: ('Итоговая · + уровень месяца', 'Final · + month level')}
POSTHOC = ('Итоговая = M3 + общий уровень месяца + оцениваемое сжатие месячных эффектов (L4); её структура выбрана после внешнего теста, оценка оптимистична.',
           'Final = M3 + shared month level + estimated shrinkage of month effects (L4); its structure was chosen after the outer test, so the scores are optimistic.')
CHANNEL_EN = {'1TV': 'Channel One', 'NTV': 'NTV', 'RUSSIA1': 'Russia 1', 'RUSSIA24': 'Russia 24'}
# Dates whose anniversaries are in the model calendar or are well-known round dates.
PEAKS = {'lenin': {'2024-01-21': (('100 лет со дня смерти\n(21.01.2024)', '100 years since death\n(21 Jan 2024)'), (8, -2)),
                   '2024-04-22': (('день рождения\n(22.04)', 'birthday\n(22 Apr)'), (8, -2)),
                   '2023-04-22': (('день рождения\n(22.04)', 'birthday\n(22 Apr)'), (8, -2))},
         'stalin': {'2023-02-02': (('80 лет Сталинградской\nпобеды (2.02.2023)', '80 years since the Battle\nof Stalingrad (2 Feb 2023)'), (6, 12)),
                    '2023-03-05': (('70 лет со дня\nсмерти (5.03.2023)', '70 years since\ndeath (5 Mar 2023)'), (8, -8)),
                    '2023-12-22': (('около дня\nрождения (21.12)', 'near the\nbirthday (21 Dec)'), (8, -2)),
                    '2025-05-08': (('канун 80-летия\nПобеды (8.05.2025)', 'eve of the 80th\nVictory Day (8 May 2025)'), (8, -2))}}


def tr(ru, en):
    """Label in the current language; also takes a (ru, en) pair."""
    return en if LANG == 'en' else ru


def dec(s):
    """Decimal comma in Russian labels."""
    return s if LANG == 'en' else s.replace('.', ',')


def person_label(person):
    return tr(fa.PERSON[person], person.capitalize())


def channel_label(channel):
    return tr(fa.LABELS[channel], CHANNEL_EN[channel])


LONG_DASHES = str.maketrans({chr(code): '-' for code in range(0x2012, 0x2016)})


def clean_figure_text(fig):
    """Normalize every visible label, including text from shared helpers."""
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_text(artist.get_text().translate(LONG_DASHES))


def save(fig, name):
    clean_figure_text(fig)
    if SLIDES is not None:
        fig.savefig(SLIDES/f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    svg = FIG/f'{name}.svg'
    svg.write_text(svg.read_text(encoding='utf-8').replace(
        "font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif"), encoding='utf-8')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    fig.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def header(fig, title, subtitle, size=22, top=.98, gap=.075):
    if SLIDES is not None:
        return
    fig.text(.065, top, title, fontsize=size, va='top')
    fig.text(.065, top-gap, subtitle, color=MUTED, fontsize=10)


def final_paths():
    with np.load(rv.OUT/f'final_paths_{FINAL}.npz') as z:
        return {key: z[key].astype(np.int64) for key in rv.KEYS}, pd.DatetimeIndex(z['dates'])


def national_daily(data):
    """Sum of 4 channels on days when all channels are observed."""
    full = data.groupby('date')['observed'].all()
    keys = [person+'_'+target for person in m.PEOPLE for target in ('mentions', 'broadcasts')]
    count = data.groupby('date').channel.nunique()
    return data.groupby('date')[keys].sum()[full & count.eq(len(m.CHANNELS))]


def national_monthly(data):
    """Monthly totals of 4 channels; months with gaps are scaled by observed channel-days."""
    d = data.assign(month=data.date.dt.to_period('M'))
    seen = d[d.observed.astype(bool)]
    totals = seen.groupby('month')[[p+'_mentions' for p in m.PEOPLE]].sum()
    cells = seen.groupby('month').size(); expected = totals.index.days_in_month*len(m.CHANNELS)
    scaled = totals.mul(expected/cells.reindex(totals.index), axis=0)
    return scaled.assign(complete=(cells.reindex(totals.index).to_numpy() == expected), share=cells/expected)


# 1. History with labelled peaks.
def history_figure(data):
    nat = national_daily(data)
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 7.6), sharex=True)
    header(fig, tr('Упоминания Ленина и Сталина за 2023-2026', 'Mentions of Lenin and Stalin, 2023-2026'),
           tr('Сумма четырёх каналов за день  /  только дни, когда наблюдаются все каналы  /  линия: среднее за 28 дней',
              'Sum of four channels per day  /  only days when all channels are observed  /  line: 28-day mean'), top=.985, gap=.05)
    for ax, person in zip(axes, m.PEOPLE):
        s = nat[person+'_mentions']; color = fa.COLORS[person]
        gaps = s.reindex(pd.date_range(s.index.min(), s.index.max()))  # incomplete days stay gaps
        ax.plot(gaps.index, gaps, color=color, lw=.6, alpha=.45)
        roll = gaps.rolling(28, min_periods=14, center=True).mean()
        ax.plot(roll.index, roll, color=color, lw=2)
        for day, (label, offset) in PEAKS[person].items():
            t = pd.Timestamp(day); y = s.get(t, np.nan)
            if np.isnan(y):
                continue
            ax.scatter([t], [y], s=26, color=INK, zorder=4)
            ax.annotate(tr(*label), (t, y), xytext=offset, textcoords='offset points', fontsize=8.5, color=INK,
                        va='top' if offset[1] < 0 else 'bottom')
        near = np.zeros(len(s), bool)
        for day in PEAKS[person]:
            near |= np.abs((s.index-pd.Timestamp(day)).days) <= 3
        others = s[~near].nlargest(2)
        for t, y in others.items():
            ax.annotate(tr(f'{t:%d.%m.%Y}', f'{t.day} {t:%b %Y}'), (t, y), xytext=(6, -2), textcoords='offset points', fontsize=8, color=MUTED, va='top')
        ax.set_title(person_label(person), loc='left', fontsize=13, pad=8)
        ax.set_ylim(0, s.max()*1.12); ax.yaxis.set_major_locator(MaxNLocator(4)); fa.clean(ax)
        ax.set_ylabel(tr('упоминаний в день', 'mentions per day'), fontsize=9)
    axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.subplots_adjust(top=.87, bottom=.08, left=.07, right=.985, hspace=.28)
    fig.text(.065, .015, tr('Подписаны годовщины из календаря модели; серым - крупнейшие пики без календарной даты. '
                            'Совпадение с датой не означает причину.',
                            'Labelled: anniversaries from the model calendar; grey: the largest peaks without a calendar date. '
                            'A coinciding date does not imply a cause.'), color=MUTED, fontsize=9)
    save(fig, 'history_peaks')


# 2. Monthly history and the forecast on one axis.
def history_forecast_figure(data, month):
    hist = national_monthly(data); hist = hist[hist.index <= pd.Period('2026-09', 'M')]; x = hist.index.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 7.4), sharex=True)
    header(fig, tr('История и прогноз на месяц', 'History and the monthly forecast'),
           tr('Упоминания, сумма четырёх каналов за месяц  /  прогноз на 3 октября - 2 ноября 2026: '
              'точка - среднее, толстая линия - 80%, тонкая - 95%',
              'Mentions, sum of four channels per month  /  forecast for 3 October - 2 November 2026: '
              'dot - mean, thick line - 80%, thin - 95%'), top=.985, gap=.05)
    for ax, person in zip(axes, m.PEOPLE):
        key = person+'_mentions'; color = fa.COLORS[person]; ok = hist.complete.to_numpy()
        ax.plot(x, hist[key], color=LIGHT, lw=1.2)
        ax.scatter(x[ok], hist[key][ok], s=22, color=INK, zorder=3)
        ax.scatter(x[~ok], hist[key][~ok], s=22, color=PAPER, edgecolor=INK, linewidth=1, zorder=3)
        octs = x[x.month == 10]
        for t in octs:
            ax.axvspan(t-pd.Timedelta(days=14), t+pd.Timedelta(days=14), color=GRID, alpha=.6, lw=0)
        r = month[(month.version == 'v2_final') & (month.channel == 'ALL') & (month.target == key)].iloc[0]
        t = pd.Timestamp('2026-10-01')
        ax.axvspan(t-pd.Timedelta(days=14), t+pd.Timedelta(days=14), color=color, alpha=.10, lw=0)
        ax.plot([t, t], [r.lo95, r.hi95], color=color, lw=1.3)
        ax.plot([t, t], [r.lo80, r.hi80], color=color, lw=6, solid_capstyle='butt', alpha=.8)
        ax.scatter([t], [r['mean']], s=46, color=color, zorder=4, edgecolor=PAPER, linewidth=1)
        ax.annotate(tr('прогноз 3.10-2.11', 'forecast 3 Oct - 2 Nov')+f'\n{r["mean"]:.0f} [{r.lo80:.0f}; {r.hi80:.0f}]', (t, r.hi95), xytext=(-6, 4),
                    textcoords='offset points', ha='right', fontsize=9, color=color)
        ax.set_title(person_label(person), loc='left', fontsize=13, pad=8)
        ax.set_ylim(0, max(hist[key].max(), r.hi95)*1.12); ax.yaxis.set_major_locator(MaxNLocator(5)); fa.clean(ax)
        ax.set_ylabel(tr('упоминаний за месяц', 'mentions per month'), fontsize=9)
    axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].set_xlim(pd.Timestamp('2022-12-10'), pd.Timestamp('2026-11-01'))
    fig.subplots_adjust(top=.87, bottom=.09, left=.07, right=.985, hspace=.28)
    fig.text(.065, .015, tr('● полный месяц   ○ в месяце есть пропущенные дни канала: сумма пересчитана на полный месяц   '
                            'серые полосы - октябри прошлых лет',
                            '● complete month   ○ month with missing channel-days: total rescaled to the full month   '
                            'grey bands - Octobers of past years'), color=MUTED, fontsize=9)
    save(fig, 'history_forecast')


# 3. What the model learned: calendar effects.
def calendar_figure(data, source_sha):
    fits = rv.v2_fits(data, rj.BASE_SEED+99999+1000, FINAL, source_sha)  # the cached final posterior
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4), gridspec_kw={'width_ratios': [1.6, 1, 1.25]})
    header(fig, tr('Чему научилась модель: календарь', 'What the model learned: the calendar'),
           tr('Во сколько раз меняются шансы, что запись упомянет фигуру  /  апостериорная медиана, 80% и 95%  /  1 = обычный день',
              'Factor by which the odds that a record mentions the figure change  /  posterior median, 80% and 95%  /  1 = ordinary day'),
           top=.985, gap=.05)
    holidays = [('lenin_death', tr('смерть Ленина 21.01', 'Lenin death 21 Jan')), ('lenin_birth', tr('рожд. Ленина 22.04', 'Lenin birth 22 Apr')),
                ('stalin_death', tr('смерть Сталина 5.03', 'Stalin death 5 Mar')),
                ('stalin_official_birth', tr('рожд. Сталина 21.12', 'Stalin birth 21 Dec')),
                ('may9', tr('9 мая', '9 May')), ('may8', tr('8 мая', '8 May')), ('nov7', tr('7 ноября', '7 Nov'))]
    for row, person in enumerate(m.PEOPLE):
        fit = fits['B_'+person]; theta = fit['theta']; names = fit['names']; color = fa.COLORS[person]
        col = lambda prefix: theta[:, [names.index(n) for n in names if n.startswith(prefix)]]
        month = col('month_c')@helmert(12, full=False); weekday = col('weekday_c')@helmert(7, full=False)
        exact = theta[:, [names.index('exact_'+h) for h, _ in holidays]]
        for ax, values, labels in ((axes[row, 0], month, tr(*MONTHS)), (axes[row, 1], weekday, tr(*WEEKDAYS))):
            q = np.exp(np.quantile(values, [.025, .1, .5, .9, .975], axis=0)); xs = np.arange(len(labels))
            ax.vlines(xs, q[0], q[4], color=color, lw=1, alpha=.6); ax.vlines(xs, q[1], q[3], color=color, lw=4, alpha=.5)
            ax.scatter(xs, q[2], s=22, color=color, zorder=3)
            ax.axhline(1, color=MUTED, lw=.7, ls=(0, (3, 4))); ax.set_yscale('log')
            ax.set_xticks(xs, labels, fontsize=8.5); fa.clean(ax)
        q = np.exp(np.quantile(exact, [.025, .1, .5, .9, .975], axis=0)); ys = np.arange(len(holidays))[::-1]
        ax = axes[row, 2]
        ax.hlines(ys, q[0], q[4], color=color, lw=1, alpha=.6); ax.hlines(ys, q[1], q[3], color=color, lw=4, alpha=.5)
        ax.scatter(q[2], ys, s=22, color=color, zorder=3); ax.axvline(1, color=MUTED, lw=.7, ls=(0, (3, 4)))
        ax.set_xscale('log'); ax.set_yticks(ys, [label for _, label in holidays], fontsize=8.5); fa.clean(ax, 'x')
        axes[row, 0].set_ylabel(person_label(person), fontsize=13, color=color, rotation=0, ha='right', va='center', labelpad=14)
    for ax, title in zip(axes[0], tr(('Месяц года', 'День недели', 'Сама дата годовщины'), ('Month of year', 'Day of week', 'Anniversary date'))):
        ax.set_title(title, loc='left', fontsize=12, pad=10)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: dec(f'×{v:g}'))
    for ax in axes.flat:
        axis = ax.yaxis if ax.get_yscale() == 'log' else ax.xaxis
        ticks = [.5, .8, 1, 1.25, 1.5, 2] if axis is ax.yaxis else [.25, .5, 1, 2, 4, 8]
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks)); axis.set_major_formatter(fmt)
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.subplots_adjust(top=.87, bottom=.1, left=.1, right=.985, hspace=.35, wspace=.55)
    fig.text(.065, .02, tr('Компонента «запись упоминает фигуру» итоговой модели (бета-биномиальная, логит). Месячные эффекты сжаты к общему '
                           'уровню (L4). Коэффициенты условные, не причинные.',
                           'Component "a record mentions the figure" of the final model (beta-binomial, logit). Month effects are shrunk '
                           'towards a common level (L4). Coefficients are conditional, not causal.'), color=MUTED, fontsize=9)
    save(fig, 'calendar_effects')


# 4. Model scheme.
def scheme_figure():
    fig, ax = plt.subplots(figsize=(13.4, 6.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, top=.86, bottom=.02)
    header(fig, tr('Как устроена модель', 'How the model works'),
           tr('Для каждого канала и дня, отдельно для Ленина и Сталина', 'For each channel and day, separately for Lenin and Stalin'),
           top=.985, gap=.065)

    def box(x, y, w, h, title, body, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fill, ec=color, lw=1.4))
        ax.text(x+1.5, y+h-1.8, title, fontsize=11.5, color=INK, va='top', weight='bold')
        ax.text(x+1.5, y+h-7.0, body, fontsize=9.5, color=INK, va='top', linespacing=1.5)

    def arrow(a, b, text=''):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3, mutation_scale=14))
        if text:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+1.3, text, ha='center', fontsize=9, color=MUTED)
    box(3, 46, 94, 11, tr('Линейная часть каждой компоненты (на шкале логарифма или логита)',
                          'Linear part of each component (on the log or logit scale)'),
        tr('канал  +  тренд  +  месяц года  +  день недели  +  годовщины  +  общий уровень месяца u',
           'channel  +  trend  +  month of year  +  day of week  +  anniversaries  +  shared month level u'), INK, PAPER)
    cols = ((3, tr('N - все записи', 'N - all records'),
             tr('архивные передачи канала\nза день\n\nотрицательное биномиальное', 'archived broadcasts of the channel\nper day\n\nnegative binomial'),
             MUTED, '#F1F2EF'),
            (36, tr('B - записи с упоминанием', 'B - records with a mention'),
             tr('сколько из N записей\nупоминают фигуру\n\nбета-биномиальное', 'how many of the N records\nmention the figure\n\nbeta-binomial'),
             fa.COLORS['stalin'], '#E6F0EF'),
            (69, tr('C - все упоминания', 'C - all mentions'),
             tr('C = B + E, где E - повторные\nупоминания в тех же записях\n\nотрицательное биномиальное',
                'C = B + E, where E counts repeated\nmentions in the same records\n\nnegative binomial'),
             fa.COLORS['lenin'], '#F5ECE7'))
    for x, title, body, color, fill in cols:
        box(x, 19, 28, 20, title, body, color, fill); arrow((x+14, 45.5), (x+14, 39.8))
    arrow((31.5, 29), (35.5, 29)); arrow((64.5, 29), (68.5, 29))
    box(3, 1, 94, 13, tr('Что даёт такая схема', 'What this design gives'),
        tr('•  Траектории согласованы: B ≤ N, B ≤ C, нет упоминаний без записи (L2).\n'
           '•  Месячные эффекты сжимаются к общему уровню, сила сжатия оценивается по данным; уровень месяца u общий для каналов и дней (L4).\n'
           '•  Апостериорное распределение - HMC (L3); уже наблюдённые 1-2 октября сдвигают уровень октября (L5).',
           '•  Trajectories are consistent: B ≤ N, B ≤ C, no mentions without a record (L2).\n'
           '•  Month effects shrink towards a common level with a strength estimated from data; the month level u is shared by channels and days (L4).\n'
           '•  Posterior via HMC (L3); the already observed 1-2 October shift the October level (L5).'), LIGHT, PAPER)
    save(fig, 'model_scheme')


# 0. Quality of the final model and its competitors.
def quality_figure(daily):
    table = daily.groupby('model').agg(D=('scaled_crps', 'mean'), cov80=('coverage80', 'mean'), cov95=('coverage95', 'mean'))
    by_origin = daily.groupby(['origin', 'model']).scaled_crps.sum().unstack()[LADDER]
    counts = daily.groupby(['origin', 'model']).size().unstack()[LADDER].to_numpy()
    rng = np.random.default_rng(33)
    blocks = rng.integers(len(by_origin), size=(2000, 7))  # circular blocks of 3 months, as in v1
    idx = ((blocks[:, :, None]+np.arange(3)) % len(by_origin)).reshape(2000, 21)
    boot = by_origin.to_numpy()[idx].sum(axis=1)/counts[idx].sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), gridspec_kw={'width_ratios': [1.5, 1, 1]})
    header(fig, tr('Качество ежедневных прогнозов', 'Quality of daily forecasts'),
           tr('21 месяц вне периода настройки  /  январь 2025 - сентябрь 2026  /  одинаковые наблюдения для всех моделей',
              '21 months outside the tuning period  /  January 2025 - September 2026  /  same observations for all models'), size=23, gap=.08)
    y = {k: len(LADDER)-1-i for i, k in enumerate(LADDER)}
    for i, k in enumerate(LADDER):
        color = INK if k == FINAL else LIGHT
        skill = 1-table.loc[k, 'D']/table.loc['M1', 'D']; ci = np.quantile(1-boot[:, i]/boot[:, 0], [.025, .975])
        axes[0].plot(ci, [y[k]]*2, color=color, lw=2, alpha=.5)
        axes[0].scatter(skill, y[k], s=50 if k == FINAL else 25, color=color, zorder=3)
        if k == FINAL:
            axes[0].annotate(dec(f'−{skill:.1%}')+' CRPS', (skill, y[k]), xytext=(0, 9), textcoords='offset points',
                             ha='center', fontsize=9)
    axes[0].axvline(0, color=MUTED, lw=.7, ls=(0, (3, 4)))
    axes[0].set_yticks([y[k] for k in LADDER], [tr(*LADDER_LABEL[k]) for k in LADDER], fontsize=9)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); axes[0].set_title(tr('Улучшение CRPS к M1', 'CRPS improvement over M1'), loc='left', fontsize=12, pad=14)
    axes[0].set_xlabel(tr('правее = меньше ошибка', 'right = smaller error'), fontsize=9); fa.clean(axes[0], 'x')
    for ax, col, nominal in zip(axes[1:], ('cov80', 'cov95'), (.8, .95)):
        for k in LADDER:
            ax.scatter(table.loc[k, col], y[k], s=50 if k == FINAL else 25, color=INK if k == FINAL else LIGHT)
        ax.axvline(nominal, color=fa.COLORS['stalin'], lw=1, ls=(0, (3, 4)))
        ax.set_title(tr('Покрытие', 'Coverage')+f' {nominal:.0%}', loc='left', fontsize=12, pad=14)
        ax.set_yticks([y[k] for k in LADDER], [k if k != FINAL else tr('Итог.', 'Final') for k in LADDER])
        ax.xaxis.set_major_locator(MultipleLocator(.05)); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlim(max(0, min(nominal, table[col].min())-.05), 1.01)
        fa.clean(ax, 'x'); ax.set_xlabel(tr('доля дней в интервале', 'share of days inside the interval'), fontsize=9)
    for ax in axes:
        ax.set_ylim(-.45, len(LADDER)-.5)
    fig.subplots_adjust(top=.73, bottom=.2, left=.2, right=.98, wspace=.5)
    fig.text(.065, .025, tr('Полосы слева: 95% блочный bootstrap (блок 3 месяца).', 'Bars on the left: 95% block bootstrap (3-month blocks).')
             +'\n'+tr(*POSTHOC), color=MUTED, fontsize=9)
    save(fig, 'backtest_quality')
    return table


def calibration_figure(daily):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.8))
    header(fig, tr('Как модель описывает неопределённость', 'How the model describes uncertainty'), '', size=20)
    for ax, target, title in zip(axes, ('mentions', 'broadcasts'), tr(('Количество упоминаний', 'Количество записей с упоминанием'),
                                                                      ('Number of mentions', 'Number of records with a mention'))):
        vals = daily[(daily.model == FINAL) & daily.target.str.endswith(target)].pit
        h, bins = np.histogram(vals, bins=np.linspace(0, 1, 11))
        ax.bar((bins[1:]+bins[:-1])/2, h/len(vals), width=.075, color=fa.COLORS['stalin'], alpha=.7, linewidth=0)
        ax.axhline(.1, color=INK, lw=.8, ls=(0, (3, 4)))
        ax.set_title(title, loc='left', fontsize=12); ax.set_xlim(0, 1)
        ax.set_xlabel(tr('рандомизированный PIT', 'randomized PIT'), fontsize=9); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); fa.clean(ax)
    fig.subplots_adjust(top=.72, bottom=.22, left=.065, right=.98, wspace=.22)
    fig.text(.065, .025, tr('Итоговая модель, внешний тест. При калибровке ожидается около 10% исходов в каждом столбце (L1).',
                            'Final model, outer test. Under calibration about 10% of outcomes are expected in each bar (L1).'),
             color=MUTED, fontsize=9)
    save(fig, 'calibration')


def monthly_coverage_figure(daily, monthly):
    table = e.metrics(daily, monthly)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9))
    n = int(table.loc[FINAL, 'n_monthly'])
    header(fig, tr('Месячные интервалы итоговой модели', 'Monthly intervals of the final model'),
           tr(f'Внешний тест: {n} полных месячных итогов (16 рядов × месяцы)  /  доля итогов внутри интервала и CRPS месячной суммы',
              f'Outer test: {n} complete monthly totals (16 series × months)  /  share of totals inside the interval and CRPS of the monthly total'),
           size=21, gap=.085)
    y = {k: len(LADDER)-1-i for i, k in enumerate(LADDER)}
    for ax, (col, title, nominal) in zip(axes, (('monthly_cov80', tr('Интервал 80%', '80% interval'), .8),
                                                ('monthly_cov95', tr('Интервал 95%', '95% interval'), .95),
                                                ('Mo', tr('Ошибка месячной суммы', 'Error of the monthly total'), None))):
        for k in LADDER:
            ax.scatter(table.loc[k, col], y[k], s=60 if k == FINAL else 30, color=INK if k == FINAL else LIGHT, zorder=3)
        v = table.loc[FINAL, col]
        ax.annotate(dec(f'{v:.1%}' if nominal else f'{v:.3f}'), (v, y[FINAL]), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=9)
        if nominal:
            ax.axvline(nominal, color=fa.COLORS['stalin'], lw=1, ls=(0, (3, 4)))
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xlim(table[col][LADDER].min()-.04, 1.0); ax.set_xlabel(tr('фактическое покрытие', 'actual coverage'), fontsize=9)
        else:
            ax.set_xlim(table[col][LADDER].min()*.93, table[col][LADDER].max()*1.04); ax.set_xlabel(tr('левее = меньше ошибка', 'left = smaller error'), fontsize=9)
        ax.set_title(title, loc='left', fontsize=12, pad=14)
        ax.set_yticks([y[k] for k in LADDER], [tr(*LADDER_LABEL[k]) if ax is axes[0] else '' for k in LADDER], fontsize=9)
        ax.set_ylim(-.5, len(LADDER)-.4); fa.clean(ax, 'x')
    fig.subplots_adjust(top=.72, bottom=.22, left=.17, right=.985, wspace=.18)
    fig.text(.065, .025, tr('Пунктир - заявленный уровень. Общий для каналов уровень месяца расширяет месячные интервалы до номинала.',
                            'Dashed line - nominal level. The month level shared by channels widens monthly intervals to the nominal level.')
             +'\n'+tr(*POSTHOC), color=MUTED, fontsize=9)
    save(fig, 'monthly_coverage')
    return table


def national_test(data):
    """Sum of 4 channels per test month over observed channel-days (gaps are not zeros)."""
    rows = []
    for end in rj.TEST_ENDS:
        dates = pd.date_range(end+pd.Timedelta(days=1), end+pd.offsets.MonthEnd(1))
        actual = data[data.date.isin(dates)].set_index(['date', 'channel'])
        seen = actual.observed.unstack().reindex(index=dates, columns=m.CHANNELS).fillna(False).to_numpy(bool)
        paths = rv.load_paths('test', str(end.date()), FINAL)
        for person in m.PEOPLE:
            key = person+'_mentions'; total = (paths[key]*seen).sum(axis=(1, 2))
            y = actual[key].unstack().reindex(index=dates, columns=m.CHANNELS).to_numpy(float)
            q = np.quantile(total, [.025, .1, .9, .975])
            rows.append(dict(month=dates[0], person=person, mean=total.mean(), lo95=q[0], lo80=q[1], hi80=q[2], hi95=q[3],
                             actual=float(y[seen].sum()), missing=int((~seen).sum())))
    return pd.DataFrame(rows)


def forecast_vs_actual_figure(data):
    t = national_test(data); months = pd.DatetimeIndex(sorted(t.month.unique())); x = np.arange(len(months))
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 8.0), sharex=True)
    header(fig, tr('Прогноз месяца против факта', 'Monthly forecast versus outcome'),
           tr('Внешний тест: январь 2025 - сентябрь 2026  /  упоминания, сумма четырёх каналов  /  '
              'прогноз с начала месяца  /  толстая линия: 80%, тонкая: 95%',
              'Outer test: January 2025 - September 2026  /  mentions, sum of four channels  /  '
              'forecast from the start of the month  /  thick line: 80%, thin: 95%'), size=23, top=.985, gap=.05)
    for ax, person in zip(axes, m.PEOPLE):
        g = t[t.person == person].sort_values('month'); color = fa.COLORS[person]
        ax.vlines(x, g.lo95, g.hi95, color=color, lw=1.1, alpha=.75); ax.vlines(x, g.lo80, g.hi80, color=color, lw=7, alpha=.45)
        full = (g.missing == 0).to_numpy()
        ax.scatter(x[full], g.actual[full], s=34, color=INK, zorder=4, edgecolor=PAPER, linewidth=.8)
        ax.scatter(x[~full], g.actual[~full], s=30, color=PAPER, zorder=4, edgecolor=INK, linewidth=1.2)
        in80 = ((g.lo80 <= g.actual) & (g.actual <= g.hi80)).sum(); in95 = ((g.lo95 <= g.actual) & (g.actual <= g.hi95)).sum()
        ax.set_title(tr(f'{fa.PERSON[person]}  ·  факт внутри 80%: {in80} из {len(g)},  внутри 95%: {in95} из {len(g)}',
                        f'{person_label(person)}  ·  outcome inside 80%: {in80} of {len(g)},  inside 95%: {in95} of {len(g)}'),
                     loc='left', fontsize=12.5, pad=10)
        ax.set_ylim(bottom=0); ax.yaxis.set_major_locator(MaxNLocator(5)); fa.clean(ax)
        ax.set_ylabel(tr('упоминаний за месяц', 'mentions per month'), fontsize=9)
    names = tr(*MONTHS)
    labels = [f'{names[d.month-1]}\n{d.year}' if d.month == 1 or i == 0 else names[d.month-1] for i, d in enumerate(months)]
    axes[1].set_xticks(x, labels, fontsize=9); axes[1].set_xlim(-.6, len(x)-.4)
    fig.subplots_adjust(top=.86, bottom=.12, left=.08, right=.985, hspace=.32)
    fig.text(.065, .02, tr('● факт   ○ в месяце есть пропущенные дни канала: прогноз и факт суммируются только по наблюдённым дням.\n'
                           'Итоговая модель; её структура выбрана после этого теста, поэтому совпадение оптимистично.',
                           '● outcome   ○ month with missing channel-days: forecast and outcome are summed over observed days only.\n'
                           'Final model; its structure was chosen after this test, so the agreement is optimistic.'), color=MUTED, fontsize=9)
    save(fig, 'forecast_vs_actual')
    return t


def month_overview_figure(month):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    header(fig, tr('Прогноз на месяц по каналам', 'Monthly forecast by channel'),
           tr('3 октября - 2 ноября 2026  /  точка: среднее  /  толстая линия: 80%  /  тонкая: 95%',
              '3 October - 2 November 2026  /  dot: mean  /  thick line: 80%  /  thin: 95%'), size=24, gap=.08)
    f = month[month.version == 'v2_final']
    for ax, target, title in zip(axes, ('mentions', 'broadcasts'), tr(('Все упоминания', 'Записи с ≥1 упоминанием'),
                                                                      ('All mentions', 'Records with ≥1 mention'))):
        for c, channel in enumerate(m.CHANNELS):
            for j, person in enumerate(m.PEOPLE):
                r = f[(f.channel == channel) & (f.target == person+'_'+target)].iloc[0]
                yy = 3-c+(.12 if j == 0 else -.12); color = fa.COLORS[person]
                ax.plot([r.lo95, r.hi95], [yy, yy], color=color, lw=1, alpha=.6)
                ax.plot([r.lo80, r.hi80], [yy, yy], color=color, lw=4, solid_capstyle='round', alpha=.4)
                ax.scatter([r['mean']], [yy], s=38, color=color, zorder=4, edgecolor=PAPER, linewidth=.8)
        ax.set_title(title, loc='left', pad=15, fontsize=13)
        ax.set_yticks(range(4), [channel_label(c) for c in reversed(m.CHANNELS)])
        ax.set_ylim(-.5, 3.5); ax.set_xlim(left=0); fa.clean(ax, 'x')
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.set_xlabel(tr('сумма за 31 день', 'total over 31 days'), fontsize=9)
    fig.text(.065, .035, '● '+person_label('lenin'), color=fa.COLORS['lenin'], fontsize=11)
    fig.text(.17, .035, '● '+person_label('stalin'), color=fa.COLORS['stalin'], fontsize=11)
    fig.text(.31, .035, tr('Интервалы сумм рассчитаны по целым совместным траекториям.',
                           'Intervals of totals are computed from whole joint trajectories.'), color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.73, bottom=.17, left=.09, right=.985, wspace=.34)
    save(fig, 'month_overview')


def daily_figure(data, paths, dates, target):
    fig, axes = plt.subplots(4, 2, figsize=(13.3, 10.7))
    header(fig, tr('Упоминания по дням', 'Mentions by day') if target == 'mentions' else tr('Архивные записи с упоминанием', 'Archived records with a mention'),
           tr('История до 2 октября 2026  /  прогноз на 31 день  /  линия: среднее, полосы: 80% и 95%',
              'History up to 2 October 2026  /  31-day forecast  /  line: mean, bands: 80% and 95%'), size=23, top=.985, gap=.035)
    start, end = pd.Timestamp('2026-08-15'), pd.Timestamp('2026-11-02')
    for c, channel in enumerate(m.CHANNELS):
        hist = data[(data.channel == channel) & (data.date >= start)]
        for j, person in enumerate(m.PEOPLE):
            ax = axes[c, j]; color = fa.COLORS[person]; key = person+'_'+target
            h = hist[key].where(hist.observed.astype(bool))
            ax.plot(hist.date, h, color='#A9B1B2', lw=.8, alpha=.9); ax.scatter(hist.date, h, color='#A9B1B2', s=4, linewidths=0)
            s = paths[key][:, :, c]; q = np.quantile(s, [.025, .1, .9, .975], axis=0)
            ax.fill_between(dates, q[0], q[3], color=color, alpha=.09, linewidth=0)
            ax.fill_between(dates, q[1], q[2], color=color, alpha=.19, linewidth=0)
            ax.plot(dates, s.mean(axis=0), color=color, lw=1.8)
            ax.axvline(pd.Timestamp('2026-10-02 12:00'), color=MUTED, lw=.7, ls=(0, (3, 4)))
            ax.set_title(f'{channel_label(channel)}  ·  {person_label(person)}', loc='left', fontsize=12, pad=11)
            ax.set_xlim(start, end); ax.set_ylim(bottom=0); fa.clean(ax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=2))
            ax.set_xticks(pd.to_datetime(['2026-08-15', '2026-09-01', '2026-10-03', '2026-11-02']), tr(['15 авг', '1 сен', '3 окт', '2 ноя'], ['15 Aug', '1 Sep', '3 Oct', '2 Nov']))
            if c < 3:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(tr('в день', 'per day'), fontsize=9)
    fig.subplots_adjust(top=.89, bottom=.06, left=.065, right=.985, hspace=.55, wspace=.20)
    fig.text(.065, .015, tr('Пробел в истории = неполный день. Полосы описывают будущие наблюдаемые счётчики, включая дневную вариацию.',
                            'A gap in history = an incomplete day. Bands describe future observed counts, including day-to-day variation.'),
             fontsize=9, color=MUTED)
    save(fig, 'daily_'+target)


def aggregate_daily_paths(paths):
    """Sum channels within each joint draw; retain cross-channel dependence."""
    result = {}
    for key in rv.KEYS:
        draws = np.asarray(paths[key])
        if draws.ndim != 3 or draws.shape[2] != len(m.CHANNELS):
            raise ValueError(f'{key}: expected draws x days x {len(m.CHANNELS)} channels')
        result[key] = draws.sum(axis=2)
    return result


def daily_total_figure(data, paths, dates, target):
    """Daily four-channel totals for both figures, from whole joint paths."""
    totals = aggregate_daily_paths(paths)
    hist = national_daily(data).reindex(pd.date_range('2026-08-15', data.date.max()))
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.4))
    title = tr('Все упоминания по дням: сумма четырёх каналов',
               'Daily mentions: sum of four channels') if target == 'mentions' else tr(
               'Записи с упоминанием по дням: сумма четырёх каналов',
               'Daily records with a mention: sum of four channels')
    header(fig, title, tr('M3HR / история до 2 октября / прогноз 3 октября - 2 ноября 2026',
                         'M3HR / history through 2 October / forecast 3 October - 2 November 2026'),
           size=22, gap=.08)
    for ax, person in zip(axes, m.PEOPLE):
        key = person+'_'+target; color = fa.COLORS[person]
        ax.plot(hist.index, hist[key], color='#A9B1B2', lw=1, label=tr('история', 'history'))
        draws = totals[key]; q = np.quantile(draws, [.025, .1, .9, .975], axis=0)
        ax.fill_between(dates, q[0], q[3], color=color, alpha=.10, linewidth=0, label='95%')
        ax.fill_between(dates, q[1], q[2], color=color, alpha=.22, linewidth=0, label='80%')
        ax.plot(dates, draws.mean(axis=0), color=color, lw=2, label=tr('среднее', 'mean'))
        ax.axvline(pd.Timestamp('2026-10-02 12:00'), color=MUTED, lw=.8, ls=(0, (3, 4)))
        ax.set_title(person_label(person), loc='left', fontsize=14)
        ax.set_xlim(pd.Timestamp('2026-08-15'), dates.max()); ax.set_ylim(bottom=0)
        ax.set_ylabel(tr('сумма за день', 'total per day'), fontsize=10)
        ax.set_xticks(pd.to_datetime(['2026-08-15', '2026-09-01', '2026-10-03', '2026-11-02']),
                      tr(['15 авг', '1 сен', '3 окт', '2 ноя'], ['15 Aug', '1 Sep', '3 Oct', '2 Nov']))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True)); fa.clean(ax)
        ax.legend(frameon=False, loc='upper right', fontsize=9)
    fig.subplots_adjust(top=.76, bottom=.19, left=.065, right=.985, wspace=.20)
    fig.text(.065, .035, tr(
        'Каналы суммируются внутри каждой траектории, затем считаются интервалы 80% и 95%. Пропуск = неполный день архива.',
        'Channels are summed within each joint trajectory before computing 80% and 95% intervals. A gap = an incomplete archive day.'),
        fontsize=9, color=MUTED)
    save(fig, 'daily_total_'+target)


# 5. Probabilities of monthly events.
def events_figure():
    ev = pd.read_csv(rv.OUT/'forecast_events_v2.csv')
    ev = ev[ev.target.str.endswith('mentions') & (ev.channel != 'ALL')]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    header(fig, tr('Будет ли октябрь выше обычного?', 'Will October be above usual?'),
           tr('Вероятность, что упоминаний за 3.10-2.11 будет больше, чем в типичном месяце последнего года  /  после перекалибровки (L6)',
              'Probability that mentions over 3 Oct - 2 Nov exceed those of a typical month of the last year  /  after recalibration (L6)'),
           size=22, gap=.08)
    for ax, person in zip(axes, m.PEOPLE):
        color = fa.COLORS[person]
        for c, channel in enumerate(m.CHANNELS):
            yy = 3-c
            for level, marker, alpha in ((.5, 'o', 1.), (.75, 'D', .55)):
                r = ev[(ev.channel == channel) & (ev.target == person+'_mentions') & (ev.level == level)].iloc[0]
                off = .13 if level == .5 else -.13
                ax.plot([0, r.p_recalibrated], [yy+off]*2, color=color, lw=5, alpha=.35*alpha+.1, solid_capstyle='butt')
                ax.scatter(r.p_recalibrated, yy+off, marker=marker, s=34, color=color, alpha=alpha, zorder=3)
                ax.text(r.p_recalibrated+.02, yy+off, dec(f'{r.p_recalibrated:.0%}  (> {r.threshold:g})'),
                        va='center', fontsize=8.5, color=INK if level == .5 else MUTED)
        ax.axvline(.5, color=MUTED, lw=.7, ls=(0, (3, 4)))
        ax.set_yticks(range(4), [channel_label(c) for c in reversed(m.CHANNELS)]); ax.set_ylim(-.6, 3.6)
        ax.set_xlim(0, 1.08); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); fa.clean(ax, 'x')
        ax.set_title(person_label(person), loc='left', fontsize=13, pad=12)
    fig.text(.065, .06, tr('●  больше медианы месячных сумм за 12 месяцев   ◆  больше их 75%-квантиля   (в скобках - порог)',
                           '●  above the median of monthly totals over 12 months   ◆  above their 75% quantile   (threshold in brackets)'),
             color=INK, fontsize=9.5)
    fig.text(.065, .02, tr('Сырые вероятности модели сдвинуты к базовой частоте пробит-перекалибровкой Φ(a + b·Φ⁻¹(p)) по прошлым '
                           'прогнозам (L6); качество - Brier (L1).',
                           'Raw model probabilities are shifted towards the base rate by probit recalibration Φ(a + b·Φ⁻¹(p)) on past '
                           'forecasts (L6); quality - Brier score (L1).'), color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.74, bottom=.2, left=.1, right=.985, wspace=.3)
    save(fig, 'event_probabilities')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lang', choices=('ru', 'en'), default=None)
    parser.add_argument('--figures', default='all', help='Comma-separated output names, or all')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--slides-dir', type=Path, help='Headerless PNG figures for presentation')
    args = parser.parse_args()
    global LANG, FIG, SLIDES
    if args.lang is not None:
        LANG = args.lang
    if args.output_dir:
        FIG = args.output_dir
    if args.slides_dir:
        SLIDES = args.slides_dir
        SLIDES.mkdir(parents=True, exist_ok=True)
    fa.theme(); data, source_sha = m.load_data()
    names = {'history_peaks', 'history_forecast', 'model_scheme', 'calendar_effects',
             'backtest_quality', 'calibration', 'monthly_coverage', 'forecast_vs_actual',
             'month_overview', 'daily_mentions', 'daily_broadcasts', 'event_probabilities',
             'daily_total_mentions', 'daily_total_broadcasts'}
    requested = names if args.figures == 'all' else set(args.figures.split(','))
    if requested - names:
        parser.error('Unknown figures: '+', '.join(sorted(requested-names)))
    if 'history_peaks' in requested: history_figure(data)
    if 'model_scheme' in requested: scheme_figure()
    if 'calendar_effects' in requested: calendar_figure(data, source_sha)
    if requested & {'history_forecast', 'month_overview'}:
        month = pd.read_csv(rv.OUT/'forecast_month_v2.csv')
        if 'history_forecast' in requested: history_forecast_figure(data, month)
        if 'month_overview' in requested: month_overview_figure(month)
    if requested & {'backtest_quality', 'calibration', 'monthly_coverage'}:
        daily, monthly = e.scores('test', LADDER)
        if 'backtest_quality' in requested: quality_figure(daily)
        if 'calibration' in requested: calibration_figure(daily)
        if 'monthly_coverage' in requested: monthly_coverage_figure(daily, monthly)
    if 'forecast_vs_actual' in requested: forecast_vs_actual_figure(data)
    daily_names = {'daily_mentions', 'daily_broadcasts', 'daily_total_mentions', 'daily_total_broadcasts'}
    if requested & daily_names:
        paths, dates = final_paths()
        for target in ('mentions', 'broadcasts'):
            if 'daily_'+target in requested: daily_figure(data, paths, dates, target)
            if 'daily_total_'+target in requested: daily_total_figure(data, paths, dates, target)
    if 'event_probabilities' in requested: events_figure()


if __name__ == '__main__':
    main()
