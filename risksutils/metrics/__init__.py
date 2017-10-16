import numpy as np
from scipy.stats import norm


__all__ = ['information_value', 'information_value_binormal']


def information_value(df, feature, target, num_buck=10):
    """information value признака с целевой переменной target

    Аргументы:
      df: pandas.DataFrame
        таблица с данными
      feature: str
        признак
      trget: str
        целевая переменная
      num_buck: numeric
        количество бакетов

    Результат:
        information value: float

    Пример использования
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

    Аргументы:
      AUC: float
        Area Under Roc Curve

    Результат:
      information value: float

    Пример использования
    >>> information_value_binormal(0.5)
    0.0
    """
    return norm.ppf(auc) ** 2 * 2
