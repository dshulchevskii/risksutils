import pandas as pd
import holoviews as hv
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta
from scipy.special import logit


def woe_line(df, feature, target, num_buck=10):
    """График зависимости WoE от признака

    Аргументы:
      df: pd.DataFrame
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
    df = df[[feature, target]].dropna()

    df_agg = (
        df.assign(bucket=lambda x: pd.qcut(x[feature], q=num_buck,
                                           duplicates='drop'),
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

    # diagrams
    scatter = hv.Scatter(
        data=df_agg,
        kdims=[feature],
        vdims=['woe'],
        group='Weight of evidence',
        label=feature
    ).opts(plot=dict(show_legend=False))
    errors = hv.ErrorBars(
        data=df_agg,
        kdims=[feature],
        vdims=['woe', 'woe_u', 'woe_b'],
        group='Confident Intervals',
        label=feature
    ).opts(plot=dict(show_legend=False))
    line = hv.Curve(
        data=df_agg,
        kdims=[feature],
        vdims=['logreg'],
        group='Logistic interpolations',
        label=feature
    ).opts(plot=dict(show_legend=False))
    diagram = hv.Overlay(
        items=[scatter, errors, line],
        group='Woe line',
        label=feature
    )

    return diagram


def woe_stab(df, feature, target, date, num_buck=10, date_freq='MS'):
    """График стабильности WoE признака по времени

    Аргументы:
      df: pd.DataFrame
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

    df_agg = (
        df.loc[lambda x: x[[date, target]].notnull().all(axis=1)]
        .loc[:, [feature, target, date]]
        .assign(bucket=lambda x: pd.qcut(x[feature], q=num_buck,
                                         duplicates='drop'),
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

    data = hv.Dataset(df_agg, kdims=['bucket', date],
                      vdims=['woe', 'woe_b', 'woe_u'])
    confident_intervals = (data.to.spread(kdims=[date],
                                          vdims=['woe', 'woe_b', 'woe_u'])
                           .overlay('bucket'))
    woe_curves = (data.to.curve(kdims=[date], vdims=['woe'])
                  .overlay('bucket'))

    return confident_intervals * woe_curves


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
