import pandas as pd
import holoviews as hv
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta, norm
from scipy.special import logit
from sklearn.metrics import roc_auc_score



def woe_line(df, feature, target, num_buck=10):
    '''График зависимости WoE от признака

    Аргументы:
      df: pd.DataFrame
        таблица с данными
      feature: str
        название признака
      target: str
        название целевой переменной
      num_buck: int
        количество бакетов

    '''
    df = df.loc[:, [feature, target]].dropna()

    df_agg = (
     df.assign(score_buck=lambda df: pd.qcut(df[feature], q=num_buck, duplicates='drop'))
        # Считаем агрегаты
       .assign(application_count=1)
       .groupby('score_buck', as_index=False)
       .agg({target: 'sum', 'application_count': 'sum', feature: 'mean'})
       .rename(columns={target: 'bad_count', feature: 'feature'})
        # Общие агрегаты
       .assign(application_total=lambda df: df['application_count'].sum(),
               bad_total=lambda df: df['bad_count'].sum())
        # Производные поля
       .assign(application_rate=lambda df: df['application_count'] / df['application_total'],
               bad_rate=lambda df: df['bad_count'] / df['application_count'],
               bad_rate_total=lambda df: df['bad_total'] / df['application_total'])
        # WoE with confident intervals
       .assign(woe=lambda df: woe(df['bad_rate'], df['bad_rate_total']),
               woe_lo=lambda df: woe_ci(df['bad_count'], df['application_count'], df['bad_rate_total'])[0],
               woe_hi=lambda df: woe_ci(df['bad_count'], df['application_count'], df['bad_rate_total'])[1])
       .assign(woe_u=lambda df: df['woe_hi'] - df['woe'],
               woe_b=lambda df: df['woe'] - df['woe_lo'])
        # Оставляем только нужные поля
       .loc[:, ['feature', 'application_count', 'bad_rate', 'woe', 'woe_u', 'woe_b']]
    )

    # Logistic interpolation
    clf = LogisticRegression(C=1)
    clf.fit(df[[feature]], df[target])
    df_agg['logreg'] = logit(clf.predict_proba(df_agg[['feature']])[:, 1]) - logit(df[target].mean())

    # сonvert to holoviews data
    data = hv.Dataset(df_agg, kdims=['feature'], vdims='application_count bad_rate woe woe_u woe_b logreg'.split())
    # diagramms
    scatter = data.to.scatter('feature', ['woe', 'application_count', 'bad_rate'])
    errors = data.to.errorbars(vdims=['woe', 'woe_u', 'woe_b'])
    line = data.to.curve(vdims='logreg')

    return hv.Overlay([scatter, errors, line], group=feature)

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


def IV(score, target, num_buck=10):
    '''Compute information value

    >>> score = np.array([0, 0, 0, 2, 2, 2])
    >>> target = np.array([0, 0, 1, 0, 1, 1])
    >>> IV(score, target)
    0.4620981203732969
    '''

    return (pd.DataFrame({'score': score, 'target': target}) 
              .assign(buck = lambda df:
                  pd.qcut(df['score'], q=num_buck, duplicates='drop'))
              .assign(row_count=1)
              .groupby('buck')
              .agg({'target': 'sum', 'row_count': 'sum'})
               # Общие агрегаты
              .assign(row_total=lambda df: df['row_count'].sum(),
                      target_total=lambda df: df['target'].sum())
               # Производные поля
              .assign(row_rate=lambda df: df['row_count'] / df['row_total'],
                      target_rate=lambda df: df['target'] / df['row_count'],
                      target_rate_total=lambda df: df['target_total'] / df['row_total'])
              .assign(woe=lambda df: woe(df['target_rate'], df['target_rate_total']))
              .eval('iv = woe * ((target_rate / target_rate_total) - (1 - target_rate) / (1 - target_rate_total)) * row_rate',
                    inplace=False)
              .loc[:, 'iv']
              .sum())


def IV_binormal(AUC):
    '''Compute IV from binormal interpolation

    >>> iv = 0.15
    >>> n = 100
    >>> target = np.repeat([0, 1], n)
    >>> q = np.linspace(1/n, 1, n, endpoint=False)
    >>> score = np.hstack([
    ...     norm.ppf(q, loc=-iv/2, scale=np.sqrt(iv)),
    ...     norm.ppf(q, loc= iv/2, scale=np.sqrt(iv))]
    ... )
    >>> AUC = roc_auc_score(target, score)
    >>> AUC
    0.61020000000000008
    >>> IV_binormal(AUC)
    0.15662123293103378
    '''
    return norm.ppf(AUC) ** 2 * 2


if __name__ == '__main__':
    import doctest
    doctest.testmod()
