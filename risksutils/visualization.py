# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import wraps
from scipy.stats import beta
from scipy.special import logit
import numpy as np
import pandas as pd
import holoviews as hv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


def _set_options(func):
    """Обертка для применения визуальных настроек"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        diagramm = func(*args, **kwargs)
        for bnd, opts in [('matplotlib', matplotlib_opts),
                          ('bokeh', bokeh_opts)]:
            if (bnd in hv.Store._options  # pylint: disable=protected-access
                    and bnd == hv.Store.current_backend):
                return diagramm.opts(opts)
        return diagramm
    return wrapper


colors = hv.Cycle([  # pylint: disable=invalid-name
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


matplotlib_opts = {  # pylint: disable=invalid-name
    'Scatter.Weight_of_evidence': {                  # woe_line
        'plot': dict(show_grid=True),
    },
    'NdOverlay.Objects_rate': {                      # distribution
        'plot': dict(xrotation=45, legend_cols=1, legend_position='right'),
    },
    'Spread.Objects_rate': {                         # distribution
        'plot': dict(show_legend=True, show_grid=True),
        'style': dict(facecolor=colors),
    },
    'Overlay.Woe_Stab': {                            # woe_stab
        'plot': dict(legend_position='right'),
    },
    'Curve.Weight_of_evidence': {                    # woe_stab
        'style': dict(color=colors),
    },
    'Spread.Confident_Intervals': {                  # woe_stab
        'plot': dict(show_grid=True, xrotation=45),
        'style': dict(facecolor=colors, alpha=0.3),
    },
    'Curve.Isotonic': {                              # isotonic
        'plot': dict(show_grid=True),
    },
    'Area.Confident_Intervals': {                    # isotonic
        'style': dict(alpha=0.5),
    },
}

bokeh_opts = {  # pylint: disable=invalid-name
    'Scatter.Weight_of_evidence': {                  # woe_line
        'plot': dict(show_grid=True, tools=['hover']),
    },
    'NdOverlay.Objects_rate': {                      # distribution
        'plot': dict(xrotation=45, legend_position='right', width=450),
    },
    'Spread.Objects_rate': {                         # distribution
        'plot': dict(show_legend=True, show_grid=True, tools=['hover']),
        'style': dict(color=colors),
    },
    'Overlay.Woe_Stab': {                            # woe_stab
        'plot': dict(legend_position='right', width=450),
    },
    'Curve.Weight_of_evidence': {                    # woe_stab
        'plot': dict(tools=['hover']),
        'style': dict(color=colors),
    },
    'Spread.Confident_Intervals': {                  # woe_stab
        'plot': dict(show_grid=True, xrotation=45),
        'style': dict(color=colors, alpha=0.3),
    },
    'Curve.Isotonic': {                              # isotonic
        'plot': dict(show_grid=True, tools=['hover']),
    },
    'Area.Confident_Intervals': {                    # isotonic
        'style': dict(alpha=0.5),
    },
}


@_set_options
def woe_line(df, feature, target, num_buck=10):
    """График зависимости WoE от признака

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными
    feature : str
        название признака
    target : str
        название целевой переменной
    num_buck : int
        количество бакетов

    **Результат**

    scatter * errors * line : holoviews.Overlay
    """
    df_agg = _aggregate_data_for_woe_line(df, feature, target, num_buck)

    scatter = hv.Scatter(data=df_agg, kdims=[feature],
                         vdims=['woe'], group='Weight of evidence')
    errors = hv.ErrorBars(data=df_agg, kdims=[feature],
                          vdims=['woe', 'woe_u', 'woe_b'],
                          group='Confident Intervals')
    line = hv.Curve(data=df_agg, kdims=[feature], vdims=['logreg'],
                    group='Logistic interpolations')
    diagram = hv.Overlay(items=[scatter, errors, line],
                         group='Woe line',
                         label=feature)

    return diagram


@_set_options
def woe_stab(df, feature, target, date, num_buck=10, date_freq='MS'):
    """График стабильности WoE признака по времени

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    feature : str
        название признака

    target : str
        название целевой переменной

    date : str
        название поля со временем

    num_buck : int
        количество бакетов

    date_ferq : str
        Тип агрегации времени (по умолчанию 'MS' - начало месяца)

    **Результат**

    curves * spreads : holoviews.Overlay
    """

    df_agg = _aggregate_data_for_woe_stab(df, feature, target, date,
                                          num_buck, date_freq)

    data = hv.Dataset(df_agg, kdims=['bucket', date],
                      vdims=['woe', 'woe_b', 'woe_u'])
    confident_intervals = (data.to.spread(kdims=[date],  # pylint: disable=no-member
                                          vdims=['woe', 'woe_b', 'woe_u'],
                                          group='Confident Intervals')
                           .overlay('bucket'))
    woe_curves = (data.to.curve(kdims=[date], vdims=['woe'],  # pylint: disable=no-member
                                group='Weight of evidence')
                  .overlay('bucket'))
    diagram = hv.Overlay(items=[confident_intervals * woe_curves],
                         group='Woe Stab',
                         label=feature)
    return diagram


@_set_options
def distribution(df, feature, date, num_buck=10, date_freq='MS'):
    """График изменения распределения признака по времени

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    feature : str
        название признака

    date : str
        название поля со временем

    num_buck : int
        количество бакетов

    date_ferq : str
        Тип агрегации времени (по умолчанию 'MS' - начало месяца)

    **Результат**

    spreads : holoviews.NdOverlay
    """

    df_agg = _aggregate_data_for_distribution(df, feature, date,
                                              num_buck, date_freq)

    obj_rates = (hv.Dataset(df_agg, kdims=['bucket', date],  # pylint: disable=no-member
                            vdims=['objects_rate', 'obj_rate_l', 'obj_rate_u'])
                 .to.spread(kdims=[date],
                            vdims=['objects_rate', 'obj_rate_l', 'obj_rate_u'],
                            group='Objects rate',
                            label=feature)
                 .overlay('bucket'))

    return obj_rates


@_set_options
def isotonic(df, predict, target, calibrations_data=None):
    """Визуализация точности прогноза вероятности

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    predict : str
        прогнозная вероятность

    target : str
        бинарная (0, 1) целевая переменная

    calibrations_data : pandas.DataFrame
        таблица с калибровками

    **Результат**

    area * curve * [curve] : holoviews.Overlay
    """

    df_agg = _aggregate_data_for_isitonic(df, predict, target)

    confident_intervals = hv.Area(df_agg, kdims=['predict'],
                                  vdims=['ci_l', 'ci_h'],
                                  group='Confident Intervals')
    curve = hv.Curve(df_agg, kdims=['predict'], vdims=['isotonic'],
                     group='Isotonic')

    if calibrations_data is not None and target in calibrations_data.columns:
        calibration = hv.Curve(
            data=calibrations_data[['predict', target]].values,
            kdims=['predict'],
            vdims=['target'],
            group='Calibration',
            label='calibration'
        )
        return hv.Overlay(items=[curve, confident_intervals, calibration],
                          group='Isotonic', label=predict)
    return hv.Overlay(items=[curve, confident_intervals],
                      group='Isotonic', label=predict)


def cross_tab(df, feature1, feature2, target,
              num_buck1=10, num_buck2=10, min_sample=100,
              compute_iv=False):
    """Кросстабуляция пары признаков и бинарной целевой переменной

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    feature1 : str
        название признака 1

    feature2 : str
        название признака 2

    target : str
        название целевой переменной

    num_buck1 : int
        количество бакетов для признака 1

    num_buck2 : int
        количество бакетов для признака 2

    min_sample : int
        минимальное количество наблюдений для
        отображение доли целевой переменной в ячейке

    compute_iv : bool
        нужно ли рассчитывать information value для признаков

    **Результат**

    (rates, counts) : (pandas.Styler, pandas.Styler)
    """

    f1_buck, f2_buck, target = (
        df
        .loc[df[target].dropna().index]
        .reset_index()
        .pipe(lambda x: (_make_bucket(x[feature1], num_buck1),
                         _make_bucket(x[feature2], num_buck2),
                         x[target]))
    )

    f1_buck_names = dict(zip(range(len(f1_buck)), f1_buck.categories))
    f2_buck_names = dict(zip(range(len(f2_buck)), f2_buck.categories))

    rates = (pd.crosstab(f1_buck.codes, f2_buck.codes, target, aggfunc=np.mean,
                         margins=True, rownames=[feature1], colnames=[feature2])
             .rename(index=f1_buck_names, columns=f2_buck_names))

    counts = (pd.crosstab(f1_buck.codes, f2_buck.codes,
                          margins=True, rownames=[feature1], colnames=[feature2])
              .rename(index=f1_buck_names, columns=f2_buck_names))

    if compute_iv:
        information_val = _iv_for_cross_tab(rates, counts)

    rates[counts < min_sample] = np.nan

    rates, counts = _add_style_for_cross_tab(rates, counts)

    if compute_iv:
        return _TupleHTML((information_val, rates, counts))
    return _TupleHTML((rates, counts))


def _aggregate_data_for_woe_line(df, feature, target, num_buck):
    df = df[[feature, target]].dropna()

    df_agg = (
        df.assign(bucket=lambda x: _make_bucket(x[feature], num_buck),
                  obj_count=1)
        .groupby('bucket', as_index=False)
        .agg({target: 'sum', 'obj_count': 'sum', feature: 'mean'})
        .dropna()
        .rename(columns={target: 'target_count'})
        .assign(obj_total=lambda x: x['obj_count'].sum(),
                target_total=lambda x: x['target_count'].sum())
        .assign(obj_rate=lambda x: x['obj_count'] / x['obj_total'],
                target_rate=lambda x: x['target_count'] / x['obj_count'],
                target_rate_total=lambda x: x['target_total'] / x['obj_total'])
        .assign(woe=lambda x: _woe(x['target_rate'], x['target_rate_total']),
                woe_lo=lambda x: _woe_ci(x['target_count'], x['obj_count'],
                                         x['target_rate_total'])[0],
                woe_hi=lambda x: _woe_ci(x['target_count'], x['obj_count'],
                                         x['target_rate_total'])[1])
        .assign(woe_u=lambda x: x['woe_hi'] - x['woe'],
                woe_b=lambda x: x['woe'] - x['woe_lo'])
        .loc[:, [feature, 'obj_count', 'target_rate', 'woe', 'woe_u', 'woe_b']]
    )

    # Logistic interpolation
    clf = Pipeline([
        ('scalle', StandardScaler()),
        ('log_reg', LogisticRegression(C=1))
    ])
    clf.fit(df[[feature]], df[target])
    df_agg['logreg'] = _woe(clf.predict_proba(df_agg[[feature]])[:, 1],
                            np.repeat(df[target].mean(), df_agg.shape[0]))

    return df_agg


def _aggregate_data_for_woe_stab(df, feature, target,
                                 date, num_buck, date_freq):
    return (
        df.loc[lambda x: x[[date, target]].notnull().all(axis=1)]
        .loc[:, [feature, target, date]]
        .assign(bucket=lambda x: _make_bucket(x[feature], num_buck),
                obj_count=1)
        .groupby(['bucket', pd.Grouper(key=date, freq=date_freq)])
        .agg({target: 'sum', 'obj_count': 'sum'})
        .reset_index()
        .assign(
            obj_total=lambda x: (
                x.groupby(pd.Grouper(key=date, freq=date_freq))
                ['obj_count'].transform('sum')),
            target_total=lambda x: (
                x.groupby(pd.Grouper(key=date, freq=date_freq))
                [target].transform('sum')))
        .assign(obj_rate=lambda x: x['obj_count'] / x['obj_total'],
                target_rate=lambda x: x[target] / x['obj_count'],
                target_rate_total=lambda x: x['target_total'] / x['obj_total'])
        .assign(woe=lambda x: _woe(x['target_rate'], x['target_rate_total']),
                woe_lo=lambda x: _woe_ci(x[target], x['obj_count'],
                                         x['target_rate_total'])[0],
                woe_hi=lambda x: _woe_ci(x[target], x['obj_count'],
                                         x['target_rate_total'])[1])
        .assign(woe_u=lambda x: x['woe_hi'] - x['woe'],
                woe_b=lambda x: x['woe'] - x['woe_lo'])
    )


def _aggregate_data_for_distribution(df, feature, date,
                                     num_buck, date_freq):
    return (
        df.loc[:, [feature, date]]
        .assign(bucket=lambda x: _make_bucket(x[feature], num_buck),
                obj_count=1)
        .groupby(['bucket', pd.Grouper(key=date, freq=date_freq)])
        .agg({'obj_count': 'sum'})
        .pipe(lambda x:  # заполняем нулями все не появившееся значения
              x.reindex(pd.MultiIndex.from_product(x.index.levels,
                                                   names=x.index.names),
                        fill_value=0))
        .reset_index()
        .assign(
            obj_total=lambda x: (
                x.groupby(pd.Grouper(key=date, freq=date_freq))
                ['obj_count'].transform('sum')))
        .assign(obj_rate=lambda x: x['obj_count'] / x['obj_total'])
        .sort_values([date, 'bucket'])
        .reset_index(drop=True)
        .assign(objects_rate=lambda x:
                x.groupby(date).apply(
                    lambda y: y.obj_rate.cumsum().to_frame())
                .reset_index(drop=True))
        .assign(obj_rate_u=0,
                obj_rate_l=lambda x: x['obj_rate'])
    )


def _make_bucket(series, num_buck):
    bucket = np.ceil(series.rank(pct=True) * num_buck).fillna(num_buck + 1)
    bucket = pd.Categorical(bucket, categories=np.sort(bucket.unique()),
                            ordered=True)
    agg = series.groupby(bucket).agg(['min', 'max'])

    def _format_buck(row):
        if row.name == num_buck + 1:
            return 'missing'
        if row['min'] == row['max']:
            return _format_val(row['min'])
        return '[{}; {}]'.format(
            _format_val(row['min']),
            _format_val(row['max'])
        )

    names = agg.apply(_format_buck, axis=1)
    return bucket.rename_categories(names.to_dict())


def _format_val(x, precision=3):
    """format a value for _make_buck

    >>> _format_val(0.00001)
    '1e-05'
    >>> _format_val(2.00001)
    '2.0'
    >>> _format_val(1000.0)
    '1000'
    >>> _format_val('foo')
    'foo'
    """
    if isinstance(x, float):
        if np.equal(np.mod(x, 1), 0):
            return '%d' % x
        if not np.isfinite(x):
            return '%s' % x
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return '%s' % np.around(x, digits)
    return '%s' % x


def _woe(tr, tr_all):
    '''Compute Weight Of Evidence from target rates

    >>> _woe(np.array([0.1, 0, 0.1]), np.array([0.5, 0.5, 0.1]))
    array([-2.19722458, -6.90675478,  0.        ])
    '''
    tr, tr_all = np.clip([tr, tr_all], 0.001, 0.999)
    return logit(tr) - logit(tr_all)


def _woe_ci(t, cnt, tr_all, alpha=0.05):
    '''Compute confident bound for WoE'''
    tr_lo, tr_hi = _clopper_pearson(t, cnt, alpha)
    woe_lo = _woe(tr_lo, tr_all)
    woe_hi = _woe(tr_hi, tr_all)
    return woe_lo, woe_hi


def _clopper_pearson(k, n, alpha=0.32):
    """Clopper Pearson intervals are a conservative estimate

    See also
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    >>> _clopper_pearson(0, 10)
    (0.0, 0.16744679259812678)
    """
    lo = beta.ppf(alpha / 2, k, n - k + 1)
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    lo = np.nan_to_num(lo)
    hi = 1 - np.nan_to_num(1 - hi)
    return lo, hi


def _aggregate_data_for_isitonic(df, predict, target):
    """Подготавливаем данные для рисования Isotonic диаграммы"""
    reg = IsotonicRegression()
    return (df[[predict, target]]                # выбираем только два поля
            .dropna()                            # оставляем только непустые
            .rename(columns={predict: 'predict',
                             target: 'target'})  # меняем их названия
            .assign(isotonic=lambda df:          # значение прогноза IR
                    reg.fit_transform(           # обучаем и считаем прогноз.
                        X=(df['predict'] +          # 🔫IR не работает с
                           1e-7 * np.random.rand(len(df))),
                        y=df['target']           # повторяющимися значениями
                    ))                           # поэтому костыльно делаем их
            .groupby('isotonic')                 # разными.
            .agg({'target': ['sum', 'count'],    # Для каждого значения ir
                  'predict': ['min', 'max']})    # агрегируем target
            .reset_index()
            .pipe(_compute_confident_intervals)   # доверительные интервалы
            .pipe(_stack_min_max))                # Преобразуем в нужный формат


def _compute_confident_intervals(df):
    """Добавляем в таблицу доверительные интервалы"""
    df['ci_l'], df['ci_h'] = _clopper_pearson(
        k=df['target']['sum'],
        n=df['target']['count'],
        alpha=0.05,
    )
    return df


def _stack_min_max(df):
    """Перегруппировываем значения в таблице для последующего рисования"""
    stack = (df['predict']                 # predict - Мульти Индекс,
             .stack()                      # Каждой строчке сопоставляем
                                           # две строчки со значениями
             .reset_index(1, drop=True)    # для min и для max,
             .rename('predict'))           # а потом меням название поля
    df = pd.concat([stack, df['isotonic'],
                    df['ci_l'],
                    df['ci_h']], axis=1)
    df['ci_l'] = df['ci_l'].cummax()       # Делаем границы монотонными
    df['ci_h'] = df[::-1]['ci_h'].cummin()[::-1]
    return df


def _hex_to_rgb(color):
    """
    >>> _hex_to_rgb('#dead13')
    (222, 173, 19)
    """
    return tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    """
    >>> _rgb_to_hex((222, 173, 19))
    '#dead13'
    """
    return '#%02x%02x%02x' % tuple(rgb)


def _color_interpolate(values, bounds_colors):
    """Интерполируем цвет исходя из границ

    >>> _color_interpolate([1, 1.5], [(0, '#010101'), (2, '#050905')])
    bound
    0.0    #010101
    1.0    #030503
    1.5    #040704
    2.0    #050905
    dtype: object
    """
    return (
        pd.DataFrame
        .from_records(bounds_colors, columns=['bound', 'color'])
        .groupby('bound')
        .first()
        .loc[:, 'color']
        .apply(_hex_to_rgb)
        .apply(pd.Series)
        .append(pd.DataFrame(index=pd.Series(values, name='bound')))
        .sort_index()
        .interpolate('index')
        .fillna(method='ffill')
        .fillna(method='bfill')
        .astype(np.int)
        .reset_index()
        .drop_duplicates()
        .set_index('bound')
        .apply(_rgb_to_hex, axis=1)
    )


class _TupleHTML(tuple):
    def _repr_html_(self):
        return '<br>'.join(i._repr_html_()  # pylint: disable=protected-access
                           if hasattr(i, '_repr_html_')
                           else repr(i)
                           for i in self)


def _add_style_for_cross_tab(rates, counts):

    rates_colors = _color_interpolate(
        values=rates.unstack().dropna(),
        bounds_colors=[
            (rates.unstack().min(), '#63bf7a'),     # green
            (rates.unstack().median(), '#ffea84'),  # yellow
            (rates.unstack().max(), '#f7686b')      # red
        ]
    )

    counts_colors = _color_interpolate(
        values=counts.unstack().dropna(),
        bounds_colors=[
            (counts.iloc[:-1, :-1].unstack().min(), '#f2f2f2'),    # light grey
            (counts.iloc[:-1, :-1].unstack().median(), '#bfbfbf'),
            (counts.iloc[:-1, :-1].unstack().max(), '#7f7f7f')     # grey
        ]
    )

    rotate_col_heading_style = dict(
        selector="th[class*='col_heading']",
        props=[("-webkit-transform", "rotate(-45deg)"),
               ('max-width', '50px')]
    )

    rates = (
        rates.style
        .applymap(lambda x: 'background-color: %s' % rates_colors[x]
                  if x == x else '')  # pylint: disable=comparison-with-itself
        .format("{:.2%}")
        .set_table_styles([rotate_col_heading_style])
    )

    counts = (
        counts.style
        .applymap(lambda x: 'background-color: %s' % counts_colors[x]
                  if x == x else '')  # pylint: disable=comparison-with-itself
        .set_table_styles([rotate_col_heading_style])
    )

    return rates, counts


def _iv_from(rates, counts, all_rate, all_count):
    return (
        pd.DataFrame({'tr': rates, 'cnt': counts})
        .dropna()
        .assign(tr=lambda x: np.clip(x['tr'], 0.001, 0.999))
        .eval('     (    (tr/{all_tr}) -    ((1-tr)/(1-{all_tr})) )'
              '   * ( log(tr/{all_tr}) - log((1-tr)/(1-{all_tr})) )'
              '   * ( cnt/{all_cnt})'
              ''.format(all_tr=all_rate, all_cnt=all_count))
        .sum()
    )


def _iv_for_cross_tab(rates, counts):
    return (
        pd.DataFrame
        .from_records([
            (rates.index.name, _iv_from(
                rates.iloc[:-1, -1], counts.iloc[:-1, -1],
                rates.iloc[-1, -1], counts.iloc[-1, -1])),
            (rates.columns.name, _iv_from(
                rates.iloc[-1, :-1], counts.iloc[-1, :-1],
                rates.iloc[-1, -1], counts.iloc[-1, -1])),
            ('%s %s' % (rates.index.name, rates.columns.name), _iv_from(
                rates.iloc[:-1, :-1].unstack(),
                counts.iloc[:-1, :-1].unstack(),
                rates.iloc[-1, -1], counts.iloc[-1, -1]))
        ])
        .rename(columns={0: 'feature', 1: 'IV'})
        .set_index('feature')
    )


_Plot = namedtuple('Plot', ['selector', 'diagram'])


class InteractiveIsotonic():
    """Интерактивная визуализация точности прогноза вероятности

    **Аргументы**

    data : pandas.DataFrame
        таблица с данными

    pdims : list
        список названий столбцов с предсказаниями

    tdims : list
        список названий столбцов с целевыми переменными

    ddims : list
        список названий столбцов с датами

    gdims : list
        список названий столбцов с категориальными полями

    calibrations_data : pandas.DataFrame
        таблица содержащая калибровки прогноза
        в целевые переменных tdims
        должна содержать столбец c именем predict
        ::
            tdims = ['t1', 't2']
            calibrations_data = pd.DataFrame({
                'predict': [0, 0.3, 0.6],
                't1': [0, 0.1, 0.2],
                't2': [0, 0.4, 0.8]
            })
        если аргумент задан, то на диаграммах isotonic
        будут присутствовать графики калибровок

    **Результат**

    diagram
        объект с набором связанных интерактивных диаграмм

        isotonic : hv.DynamicMap
            зависимость частоты наступления события
            от прогноза. Содержит виджеты для каждого
            выбора прогноза (pdims) и для выбора целевой
            переменной (tdims)
        обращаться к диаграммам нужно, как к атрибутам
        ::
            diagram.isotonic

        доступны диаграммы для категориальных полей, указанных
        в gdims, и для временных, указанных в ddims
        обращаться к ним можно по имени, например
        ::
            ddims = ['request_dt', 'response_dt']
            diagram.request_dt
            diagram.response_dt
        на данных диаграммах можно указать подвыборку, с помощью
        виджетов диаграмм bokeh, тогда пересчитаются и все
        оставшиеся диаграммы

    """

    def __init__(self, data, pdims, tdims, ddims=None, gdims=None,
                 calibrations_data=None):
        self.data = data
        self._pdims = pdims
        self._tdims = tdims
        self._gdims = gdims if gdims else []
        self._ddims = ddims if ddims else []
        self._calibrations_data = calibrations_data
        self._check_fields()        # Проверяем форматы полей
        self._diagrams = {}         # Здесь будем хранить диаграммы
        self._make_bars_static()    # Создаем диаграммы с категориями
        self._make_area_static()    # С датами
        self._make_charts()         # Конвертим их в готовые диаграммы
        self._make_isotinic_plot()

    def _get_count(self, dim, conditions=None):

        if conditions is None:
            df = self.data
        else:
            df = self.data.loc[conditions]

        return (df.groupby(dim)
                .size()
                .reset_index()
                .rename(columns={0: 'count'}))

    def _make_bars_static(self):
        """Создаем столбчатые диаграммы с выбором категорий"""
        for dim in self._gdims:
            df = self._get_count(dim)
            diagram = (hv.Bars(df, kdims=[dim], vdims=['count'])
                       .opts(plot=dict(tools=['tap'])))
            selector = (hv.streams
                        .Selection1D(source=diagram)
                        .rename(index=dim))
            self._diagrams[dim] = _Plot(selector, diagram)

    def _make_area_static(self):
        """Создаем диаграммы с выбором диапазона дат"""
        for dim in self._ddims:
            df = self._get_count(dim)
            diagram = hv.Area(df, kdims=[dim], vdims=['count'])
            selector = (hv.streams
                        .BoundsX(source=diagram)
                        .rename(boundsx=dim))
            self._diagrams[dim] = _Plot(selector, diagram)

    def _conditions(self, **kwargs):
        """Извлекаем все условия для подвыборки из статичных диаграмм"""
        conditions = np.repeat(True, len(self.data))  # Сначала задаем True

        for dim, value in kwargs.items():           # Название всех ограничений
            if dim in self._gdims:                  # совпадает с полями
                _, diagram = self._diagrams[dim]
                categories = diagram.data.loc[value][dim]
                if not categories.empty:
                    conditions &= self.data[dim].isin(categories)
            elif dim in self._ddims:
                if value:
                    left, right = value
                    conditions &= self.data[dim].between(left, right)
        return conditions

    def _make_charts(self):
        """Создаем диаграммы вместе с меняющейся"""
        for dim in self._gdims + self._ddims:
            self.__dict__[dim] = (             # Добавляем диаграмму в атрибуты
                self._diagrams[dim].diagram *  # self. Небезопасно, если
                self._make_one_chart(dim))     # уже что-то есть с именем

    def _make_one_chart(self, dim):
        """Одна динамически обновляемая диаграмма"""

        selectors = [s for s, d in self._diagrams.values()]

        if dim in self._ddims:
            diagram_type = hv.Area
        elif dim in self._gdims:
            diagram_type = hv.Bars

        def bar_chart(**kwargs):
            data = self._get_count(dim, self._conditions(**kwargs))
            return diagram_type(data, kdims=[dim], vdims=['count'])

        return hv.DynamicMap(bar_chart, streams=selectors)

    def _make_isotinic_plot(self):
        """Создаем диаграмму с IR"""

        kdims = [hv.Dimension('predict', values=self._pdims),
                 hv.Dimension('target', values=self._tdims)]

        selectors = [s for s, d in self._diagrams.values()]

        def chart(target, predict, **kwargs):
            condisions = self._conditions(**kwargs)
            data = self.data.loc[condisions]
            return isotonic(data, predict, target, self._calibrations_data)

        iso_chart = hv.DynamicMap(chart, kdims=kdims, streams=selectors)
        self.__dict__['isotonic'] = iso_chart

    def _check_fields(self):
        """Проверяльщик формата"""
        assert isinstance(self.data, pd.DataFrame), 'data must be DataFrame'
        assert self._pdims is not None, '{} must be not None'.format('pdims')
        assert self._tdims is not None, '{} must be not None'.format('tdims')

        for dims in [self._gdims, self._tdims, self._ddims, self._pdims]:
            assert isinstance(dims, list), '{} must be list'.format(dims)
            for col in dims:
                assert col in self.data.columns, '{} must be a column of data'
