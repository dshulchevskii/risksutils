import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import interp1d


def recalibration(data, features, target, target_calibration=None,
                  calibration_data=None, offset=None, use_bias=True):

    kw = {}
    if offset:
        kw[offset] = data[offset]

    if target_calibration:
        short = calibration_data[target].values
        long = calibration_data[target_calibration].values
        family = create_family(short=short, long=long)
    else:
        family = sm.families.Binomial()

    features = features if isinstance(features, str) else " + ".join(features)
    formula = '{target} ~ {features} {use_bias}'.format(
        target=target,
        features=features,
        use_bias="" if use_bias else " - 1"
    )
    model = smf.glm(formula, data, family=family, **kw)
    model = model.fit()

    return model


class Interpolation(sm.families.links.Link):

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


class Composition(sm.families.links.Link):

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
        return (f.deriv(g(p)) * g.deriv2(p) + f.deriv2(g(p)) * g.deriv(p) ** 2)


def create_family(short, long):
    class LogitAndInterpolation(Composition):

        def __init__(self):
            interpolate = Interpolation(x=short, y=long)
            logit = sm.families.links.logit()
            super().__init__(f=logit, g=interpolate)

    class Binomial(sm.families.Binomial):
        links = safe_links = [LogitAndInterpolation]

    return Binomial(LogitAndInterpolation)
