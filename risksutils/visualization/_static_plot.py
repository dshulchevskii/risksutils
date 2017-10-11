import pandas as pd
import holoviews as hv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import beta
from scipy.special import logit


def woe_line(df, feature, target, num_buck=10):
    """График зависимости WoE от признака

    Аргументы:
      df: pandas.DataFrame
        таблица с данными
      feature: str
        название признака
      target: str
        название целевой переменной
      num_buck: int
        количество бакетов

    Результат:
      scatter * errors * line: holoviews.Overlay
    """
    df_agg = aggregate_data_for_woe_line(df, feature, target, num_buck)

    scatter = hv.Scatter(data=df_agg, kdims=[feature],
                         vdims=['woe'], group='Weight of evidence',
                         label=feature).opts(plot=dict(show_legend=False))
    errors = hv.ErrorBars(data=df_agg, kdims=[feature],
                          vdims=['woe', 'woe_u', 'woe_b'],
                          group='Confident Intervals',
                          label=feature).opts(plot=dict(show_legend=False))
    line = hv.Curve(data=df_agg, kdims=[feature], vdims=['logreg'],
                    group='Logistic interpolations',
                    label=feature).opts(plot=dict(show_legend=False))
    diagram = hv.Overlay(items=[scatter, errors, line],
                         group='Woe line',
                         label=feature)

    return diagram


def woe_stab(df, feature, target, date, num_buck=10, date_freq='MS'):
    """График стабильности WoE признака по времени

    Аргументы:
      df: pandas.DataFrame
        таблица с данными
      feature: str
        название признака
      target: str
        название целевой переменной
      date: str
        название поля со временем
      num_buck: int
        количество бакетов
      date_ferq: str
        Тип агрегации времени (по умолчанию 'MS' - начало месяца)

    Результат:
      curves * spreads: holoviews.Overlay
    """

    df_agg = aggregate_data_for_woe_stab(df, feature, target, date,
                                         num_buck, date_freq)

    data = hv.Dataset(df_agg, kdims=['bucket', date],
                      vdims=['woe', 'woe_b', 'woe_u'])
    confident_intervals = (data.to.spread(kdims=[date],
                                          vdims=['woe', 'woe_b', 'woe_u'],
                                          group='Confident Intervals',
                                          label=feature)
                           .overlay('bucket'))
    woe_curves = (data.to.curve(kdims=[date], vdims=['woe'],
                                group='Weight of evidence',
                                label=feature)
                  .overlay('bucket'))
    diagram = hv.Overlay(items=[confident_intervals * woe_curves],
                         group='Woe Stab',
                         label=feature)
    return diagram


def distribution(df, feature, date, num_buck=10, date_freq='MS'):
    """График изменения распределения признака по времени

    Аргументы:
      df: pandas.DataFrame
        таблица с данными
      feature: str
        название признака
      date: str
        название поля со временем
      num_buck: int
        количество бакетов
      date_ferq: str
        Тип агрегации времени (по умолчанию 'MS' - начало месяца)

    Результат:
      spreads: holoviews.NdOverlay
    """

    df_agg = aggregate_data_for_distribution(df, feature, date,
                                             num_buck, date_freq)

    obj_rates = (hv.Dataset(df_agg, kdims=['bucket', date],
                            vdims=['objects_rate', 'obj_rate_l', 'obj_rate_u'])
                 .to.spread(kdims=[date],
                            vdims=['objects_rate', 'obj_rate_l', 'obj_rate_u'],
                            group='Objects rate',
                            label=feature)
                 .overlay('bucket'))

    return obj_rates


def isotonic(df, predict, target, calibrations_data=None):
    """Визуализация точности прогноза вероятности

    Аргументы:
      df: pandas.DataFrame
        таблица с данными
      predict: str
        прогнозная вероятность
      target: str
        бинарная (0, 1) целевая переменная
      calibrations_data: pandas.DataFrame
        таблица с калибровками

    Результат:
        area * curve * [curve] : holoviews.Overlay
    """

    df_agg = aggregate_data_for_isitonic(df, predict, target)

    if calibrations_data is not None and target in calibrations_data.columns:
        calibration = hv.Curve(
            data=calibrations_data[['predict', target]].values,
            kdims=['predict'],
            vdims=['target'],
            group='Calibration',
            label='calibration'
        )
        show_calibration = True
    else:
        show_calibration = False

    confident_intervals = (hv.Area(df_agg, kdims=['predict'],
                                   vdims=['ci_l', 'ci_h'],
                                   group='Confident Intervals',
                                   label=predict)
                           .opts(style=dict(alpha=0.5)))
    curve = hv.Curve(df_agg, kdims=['predict'], vdims=['isotonic'],
                     group='Isotonic', label=predict)

    if show_calibration:
        return hv.Overlay(items=[curve, confident_intervals, calibration],
                          group='Isotonic', label=predict)
    return hv.Overlay(items=[curve, confident_intervals],
                      group='Isotonic', label=predict)


def aggregate_data_for_woe_line(df, feature, target, num_buck):
    df = df[[feature, target]].dropna()

    df_agg = (
        df.assign(bucket=lambda x: make_bucket(x[feature], num_buck),
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
        .assign(woe=lambda x: woe(x['target_rate'], x['target_rate_total']),
                woe_lo=lambda x: woe_ci(x['target_count'], x['obj_count'],
                                        x['target_rate_total'])[0],
                woe_hi=lambda x: woe_ci(x['target_count'], x['obj_count'],
                                        x['target_rate_total'])[1])
        .assign(woe_u=lambda x: x['woe_hi'] - x['woe'],
                woe_b=lambda x: x['woe'] - x['woe_lo'])
        .loc[:, [feature, 'obj_count', 'target_rate', 'woe', 'woe_u', 'woe_b']]
    )

    # Logistic interpolation
    clf = LogisticRegression(C=1)
    clf.fit(df[[feature]], df[target])
    df_agg['logreg'] = (
        logit(clf.predict_proba(df_agg[[feature]])[:, 1]) -
        logit(df[target].mean())
    )

    return df_agg


def aggregate_data_for_woe_stab(df, feature, target,
                                date, num_buck, date_freq):
    return (
        df.loc[lambda x: x[[date, target]].notnull().all(axis=1)]
        .loc[:, [feature, target, date]]
        .assign(bucket=lambda x: make_bucket(x[feature], num_buck),
                obj_count=1)
        .groupby(['bucket', pd.TimeGrouper(key=date, freq=date_freq)])
        .agg({target: 'sum', 'obj_count': 'sum'})
        .reset_index()
        .assign(
            obj_total=lambda x: (
                x.groupby(pd.TimeGrouper(key=date, freq=date_freq))
                ['obj_count'].transform('sum')),
            target_total=lambda x: (
                x.groupby(pd.TimeGrouper(key=date, freq=date_freq))
                [target].transform('sum')))
        .assign(obj_rate=lambda x: x['obj_count'] / x['obj_total'],
                target_rate=lambda x: x[target] / x['obj_count'],
                target_rate_total=lambda x: x['target_total'] / x['obj_total'])
        .assign(woe=lambda x: woe(x['target_rate'], x['target_rate_total']),
                woe_lo=lambda x: woe_ci(x[target], x['obj_count'],
                                        x['target_rate_total'])[0],
                woe_hi=lambda x: woe_ci(x[target], x['obj_count'],
                                        x['target_rate_total'])[1])
        .assign(woe_u=lambda x: x['woe_hi'] - x['woe'],
                woe_b=lambda x: x['woe'] - x['woe_lo'])
    )


def aggregate_data_for_distribution(df, feature, date,
                                    num_buck, date_freq):
    return (
        df.loc[:, [feature, date]]
        .assign(bucket=lambda x: make_bucket(x[feature], num_buck),
                obj_count=1)
        .groupby(['bucket', pd.TimeGrouper(key=date, freq=date_freq)])
        .agg({'obj_count': 'sum'})
        .pipe(lambda x:  # заполняем нулями все не появившееся значения
              x.reindex(pd.MultiIndex.from_product(x.index.levels,
                                                   names=x.index.names),
                        fill_value=0))
        .reset_index()
        .assign(
            obj_total=lambda x: (
                x.groupby(pd.TimeGrouper(key=date, freq=date_freq))
                ['obj_count'].transform('sum')))
        .assign(obj_rate=lambda x: x['obj_count'] / x['obj_total'])
        .sort_values([date, 'bucket'])
        .reset_index(drop=True)
        .assign(objects_rate=lambda x:
                x.groupby(date).apply(
                    lambda y: y.obj_rate.cumsum()).reset_index(drop=True))
        .assign(obj_rate_u=0,
                obj_rate_l=lambda x: x['obj_rate'])
    )


def make_bucket(series, num_bucket):
    bucket = np.ceil(series.rank(pct=True) * num_bucket).fillna(-1)
    agg = series.groupby(bucket).agg(['min', 'max'])
    names = agg['min'].astype(str).copy()
    names[agg['min'] != agg['max']] = ('[' + agg['min'].astype(str) +
                                       '; ' + agg['max'].astype(str) + ']')
    names.loc[-1] = 'missing'
    return bucket.map(names.to_dict())


def woe(tr, tr_all):
    '''Compute Weight Of Evidence from target rates

    >>> woe(np.array([0.1, 0, 0.1]), np.array([0.5, 0.5, 0.1]))
    array([-2.19722458, -6.90675478,  0.        ])
    '''
    tr, tr_all = np.clip([tr, tr_all], 0.001, 0.999)
    return logit(tr) - logit(tr_all)


def woe_ci(t, cnt, tr_all, alpha=0.05):
    '''Compute confident bound for WoE'''
    tr_lo, tr_hi = clopper_pearson(t, cnt, alpha)
    woe_lo = woe(tr_lo, tr_all)
    woe_hi = woe(tr_hi, tr_all)
    return woe_lo, woe_hi


def clopper_pearson(k, n, alpha=0.32):
    """Clopper Pearson intervals are a conservative estimate

    See also
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    >>> clopper_pearson(np.array([0, 2]), np.array([2, 2]))
    (array([ 0. ,  0.4]), array([ 0.6,  1. ]))
    """
    lo = beta.ppf(alpha / 2, k, n - k + 1)
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    lo[np.isnan(lo)] = 0
    hi[np.isnan(hi)] = 1
    return lo, hi


def aggregate_data_for_isitonic(df, predict, target):
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
            .pipe(compute_confident_intervals)   # доверительные интервалы
            .pipe(stack_min_max))                # Преобразуем в нужный формат


def compute_confident_intervals(df):
    """Добавляем в таблицу доверительные интервалы"""
    df['ci_l'], df['ci_h'] = proportion_confint(
        count=df['target']['sum'],
        nobs=df['target']['count'],
        alpha=0.05,
        method='beta'
    )
    df['ci_l'] = df['ci_l'].fillna(0)
    df['ci_h'] = df['ci_h'].fillna(1)
    return df


def stack_min_max(df):
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
