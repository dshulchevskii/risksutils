# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import statsmodels.formula.api as smf


def recalibration(df, features, target, target_calibration=None,
                  calibrations_data=None, offset=None, use_bias=True):
    """Построение логистической регрессии с калибровкой

    Обычная лог регрессия строится зависимость прогноза вероятности
    от линейной комбинации признаков в виде::
        prob = 1 / (1 + exp(- logit))
    В данной функции есть возможность добавить кусочно линейное
    преобразование в конце - calibration::
        prob = calibration(1 / (1 + exp(- logit)))

    При обучении линейной комбинации признаков для расчета logit
    можно добавить к ним снос (offset), который не будет обучаться

    **Аргументы**

    df : pandas.DataFrame
        таблица с данными

    features : list или str
        набор признаков в виде списка название полей, например
        ::
            ['f0', 'f1', 'f2', 'f3']
        или описание столбцов в виде patsy формулы, например
        ::
            'f0 + f1 + C(f2) + I(np.log(f3))'
        в данном случае f2 - будет категориальным признаком,
        а от f3 будет взят логарифм

    target : str
        название целевой переменной

    target_calibration : str
        название целевой переменной в которую нужно будет
        калибровать формулу

    calibrations_data : pandas.DataFrame
        таблица с соотношением калибровок вероятностей
        должна содержать столбцы target_calibration и target

    offset : str
        название поля для сноса

    use_bias : bool
        нужно ли обучать константу

    **Результат**

    model : statmodels.model
        для просмотра результатов нужно выполнить
        ::
            model.summary()

        коэффициенты доступны
        ::
            model.params
    """

    kwargs = {}
    if offset:
        kwargs['offset'] = df[offset]

    if target_calibration:
        short = calibrations_data[target].values
        long = calibrations_data[target_calibration].values
        family = _create_family(short=short, long=long)
    else:
        family = sm.families.Binomial()

    features = features if isinstance(features, str) else " + ".join(features)
    use_bias = "" if use_bias else " - 1"
    formula = '{target} ~ {features} {use_bias}'.format(
        target=target, features=features, use_bias=use_bias)

    model = smf.glm(formula, df, family=family, **kwargs)
    model = model.fit()

    return model


class _Interpolation(sm.families.links.Link):
    """Кусочно линейная функция в виде класса Link
       в методах определяем производные и обратные функции

       **Аргументы**

       x : np.array
       y : np.array

    """

    def __init__(self, x, y):
        self._call = interp1d(x, y)
        self.inverse = interp1d(y, x)

        grad = np.diff(y) / np.diff(x)
        self.deriv = interp1d(x, np.r_[grad, 1], kind='zero')

        grad = np.diff(x) / np.diff(y)
        self.inverse_deriv = interp1d(y, np.r_[grad, 1], kind='zero')

    def __call__(self, p):
        return self._call(p)

    def deriv2(self, p):
        return np.zeros_like(p)


class _Composition(sm.families.links.Link):
    """Композиция двух функций composition(x) = f(g(x))
       в методах определяем производные и обратные функции"""

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, p):
        return self.f(self.g(p))

    def inverse(self, z):
        return self.g.inverse(self.f.inverse(z))

    def deriv(self, p):
        return self.f.deriv(self.g(p)) * self.g.deriv(p)

    def inverse_deriv(self, z):
        f, g = self.f, self.g
        return g.inverse_deriv(f.inverse(z)) * f.inverse_deriv(z)

    def deriv2(self, p):
        f, g = self.f, self.g
        return f.deriv(g(p)) * g.deriv2(p) + f.deriv2(g(p)) * g.deriv(p) ** 2


def _create_family(short, long):
    class LogitAndInterpolation(_Composition):

        def __init__(self):
            interpolate = _Interpolation(x=short, y=long)
            logit = sm.families.links.logit()
            super(LogitAndInterpolation, self).__init__(f=logit, g=interpolate)

    class Binomial(sm.families.Binomial):
        links = safe_links = [LogitAndInterpolation]

    return Binomial(LogitAndInterpolation)
