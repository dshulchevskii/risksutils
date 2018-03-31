import pandas as pd
import numpy as np
from scipy.special import expit
from risksutils.models import recalibration, _Interpolation, _Composition
import statsmodels.api as sm


def generate_data(n):
    data = pd.DataFrame()
    data['logit_main'] = np.random.randn(n) * 0.5
    data['f0'] = np.random.binomial(1, 0.2, size=n)
    data['logit_long'] = data['logit_main'] + 0.4 * data['f0']
    data['prob_long'] = expit(data['logit_long'])
    data['prob_short'] = expit(0.8 * data['logit_long'] - 0.5)
    data['target_long'] = np.random.binomial(1, data['prob_long'])
    data['target_short'] = np.random.binomial(1, data['prob_short'])

    calibration = pd.DataFrame()
    grid = np.linspace(-10, 10, 200)
    calibration['target_long'] = expit(grid)
    calibration['target_short'] = expit(0.8 * grid - 0.5)

    return data, calibration


def test_recalibration():
    np.random.seed(42)
    data, calibration = generate_data(n=10000)
    model = recalibration(
        df=data,
        features=['f0'],
        target='target_short',
        target_calibration='target_long',
        calibrations_data=calibration,
        offset='logit_main',
        use_bias=True
    )
    assert np.allclose(model.params.values, np.r_[0, 0.4], 0.1, 0.1)


def test_offset():
    np.random.seed(42)
    data, _ = generate_data(n=10000)
    model = recalibration(
        df=data,
        features=['f0'],
        target='target_long',
        offset='logit_main',
        use_bias=True
    )
    assert np.allclose(model.params.values, np.r_[0, 0.4], 0.1, 0.1)


def test_vanil_logistic():
    np.random.seed(42)
    data, _ = generate_data(n=10000)
    model = recalibration(
        df=data,
        features=['logit_main', 'f0'],
        target='target_long',
        use_bias=True
    )
    assert np.allclose(model.params.values, np.r_[0, 1, 0.4], 0.1, 0.1)


def test_interpolation():
    x = np.r_[0, 0.5, 0.7]
    y = np.r_[1, 2, 4]

    interpolate = _Interpolation(x, y)

    assert np.allclose(interpolate(0), 1)
    assert np.allclose(interpolate(0.25), 1.5)

    assert np.allclose(interpolate.inverse(2), 0.5)
    assert np.allclose(interpolate.inverse(3), 0.6)

    assert np.allclose(interpolate.deriv(0.25), 2)
    assert np.allclose(interpolate.deriv(0.6), 10)

    assert np.allclose(interpolate.inverse_deriv(1.5), 0.5)
    assert np.allclose(interpolate.inverse_deriv(3), 0.1)

    assert np.allclose(interpolate.deriv2(0.25), 0)
    assert np.allclose(interpolate.deriv2(0.6), 0)


def test_composition():

    pow2 = sm.families.links.Power(power=2)
    pow3 = sm.families.links.Power(power=3)
    pow6 = sm.families.links.Power(power=6)

    composition = _Composition(f=pow2, g=pow3)

    assert np.allclose(composition(0.9), pow6(0.9))
    assert np.allclose(composition.inverse(0.9), pow6.inverse(0.9))
    assert np.allclose(composition.deriv(0.9), pow6.deriv(0.9))
    assert np.allclose(composition.inverse_deriv(0.3), pow6.inverse_deriv(0.3))
    assert np.allclose(composition.deriv2(0.3), pow6.deriv2(0.3))
