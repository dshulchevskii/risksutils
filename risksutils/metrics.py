# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import pandas as pd


def information_value(df, feature, target, num_buck=10):
    """information value признака с целевой переменной target

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    feature : str
        признак

    target : str
        целевая переменная

    num_buck : numeric
        количество бакетов

    **Результат**

    information value : float

    **Пример использования**

    >>> import pandas as pd
    >>> df = pd.DataFrame({'foo': [1, 1, 1, np.nan, np.nan],
    ...                    'bar': [0, 0, 1, 0, 1]})
    >>> information_value(df, 'foo', 'bar')
    0.11552453009332433
    """

    df = df.loc[df[target].dropna().index, [feature, target]]
    all_tr = df[target].mean()
    all_cnt = df.shape[0]

    return (
        df.assign(bucket=lambda x: np.ceil((x[feature].rank(pct=True) *
                                            num_buck).fillna(-1)),
                  cnt=1)
        .rename(columns={target: 'tr'})
        .groupby('bucket')
        .agg({'tr': 'mean', 'cnt': 'sum'})
        .assign(tr=lambda x: np.clip(x['tr'], 0.001, 0.999))
        .eval('     (    (tr/{all_tr}) -    ((1-tr)/(1-{all_tr})) )'
              '   * ( log(tr/{all_tr}) - log((1-tr)/(1-{all_tr})) )'
              '   * ( cnt/{all_cnt})'
              ''.format(all_tr=all_tr, all_cnt=all_cnt))
        .sum()
    )


def information_value_binormal(auc):
    """information value из бинормального приближения через AUC

    **Аргументы**

    AUC : float
        Area Under Roc Curve

    **Результат**

    information value : float

    **Пример использования**

    >>> information_value_binormal(0.5)
    0.0
    """
    return norm.ppf(auc) ** 2 * 2


def stability_index(df, feature, date, num_buck=10, date_freq='MS'):
    """Stability index для всех последовательных пар дат

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    feature : str
        признак

    date : str
        название поля со временем

    num_buck : numeric
        количество бакетов

    date_ferq : str
        Тип агрегации времени (по умолчанию 'MS' - начало месяца)

    **Результат**

    pd.Series

    **Пример использования**

    >>> df = pd.DataFrame({
    ...     'dt': pd.Series(['2000-01-01', '2000-01-01', '2000-01-01',
    ...                      '2000-02-01', '2000-02-02', '2000-02-01',
    ...                      '2000-04-02', '2000-04-03'],
    ...                     dtype='datetime64[ns]'),
    ...     'foo': ['a', 'a', np.nan, 'a', 'b', 'b', 'a', 'b']
    ... })
    >>> stability_index(df, 'foo', 'dt')
    dt
    2000-02-01    6.489979
    2000-04-01    0.115525
    Name: si, dtype: float64
    """
    return (
        df.loc[:, [feature, date]]
        .assign(bucket=lambda x: np.ceil((x[feature].rank(pct=True) * num_buck)
                                         .fillna(-1)),
                object_count=1)
        .groupby([pd.Grouper(key=date, freq=date_freq), 'bucket'])
        .agg({'object_count': 'sum'})
        .pipe(_fill_zero_missing_index_pair)
        .pipe(_compute_rates, date=date)
        .pipe(_compute_si_from_rates, date=date)
    )


def _fill_zero_missing_index_pair(df):
    """Заполняем нулями все не появившееся значения комбинаций индексов"""
    return df.reindex(
        pd.MultiIndex.from_product(df.index.levels, names=df.index.names),
        fill_value=0
    )


def _compute_rates(df, date):
    """Считаем доли объектов в каждом бакете"""
    return (
        df
        .reset_index()
        .assign(total_objects=lambda x: x.groupby(date)['object_count']
                .transform(sum))
        .eval('object_rate = object_count / total_objects', inplace=False)
        .set_index([date, 'bucket'])
        .drop('total_objects', axis=1)
    )


def _compute_si_from_rates(df, date):
    """Считаем stability index исходя из долей наблюдений"""
    all_dates = df.query('object_count > 0').reset_index(date)[date].unique()
    all_dates.sort()

    return (
        df.loc[all_dates[1:]]
        .assign(object_rate_prev=pd.Series(
            np.array(df.loc[all_dates[:-1]]['object_rate']),
            index=df.loc[all_dates[1:]].index
        ))
        .reset_index(date)
        .assign(object_rate_prev=lambda x: np.clip(x['object_rate_prev'],
                                                   0.001, 0.999),
                object_rate=lambda x: np.clip(x['object_rate'], 0.001, 0.999))
        .eval('si = (    object_rate  -     object_rate_prev )'
              '   * (log(object_rate) - log(object_rate_prev))', inplace=False)
        .groupby(date)
        ['si']
        .sum()
    )
