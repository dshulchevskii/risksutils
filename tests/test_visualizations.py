from io import StringIO
import pandas as pd
import numpy as np


from risksutils.visualization import (
    woe_stab, woe_line, InterIsoReg, distribution
)


def test_woe_line_simple():
    import holoviews as hv
    hv.extension('matplotlib')

    data = """
foo,bar
1,0
1,1
1,0
2,1
2,0
2,1
"""

    df = pd.read_csv(StringIO(data))

    graphics = woe_line(df, 'foo', 'bar', num_buck=2)

    expected_result = """:Overlay
   .Weight_of_evidence.Foo      :Scatter   [foo]   (woe)
   .Confident_Intervals.Foo     :ErrorBars   [foo]   (woe,woe_u,woe_b)
   .Logistic_interpolations.Foo :Curve   [foo]   (logreg)"""

    assert repr(graphics) == expected_result

    scatter, _, _ = graphics
    assert repr(scatter.data.woe.values) == 'array([-0.69314718,  0.69314718])'


def test_woe_stab_simple():
    data = """
foo,bar,dt
a,0,2015-1-1
a,1,2015-1-1
a,0,2015-1-1
, 1,2015-1-1
, 0,2015-1-1
, 1,2015-1-1
a,0,2015-2-1
a,1,2015-2-1
a,0,2015-2-1
, 1,2015-2-1
, 0,2015-2-1
, 1,2015-2-1
"""
    df = pd.read_csv(StringIO(data), parse_dates=['dt'])

    graphics = woe_stab(df, 'foo', 'bar', 'dt', num_buck=2)

    expected_result = """:Overlay
   .Confident_Intervals.Foo :NdOverlay   [bucket]
      :Spread   [dt]   (woe,woe_b,woe_u)
   .Weight_of_evidence.Foo  :NdOverlay   [bucket]
      :Curve   [dt]   (woe)"""

    assert repr(graphics) == expected_result

    _, woe_curves = graphics
    assert (repr(woe_curves.table().data['woe'].values) ==
            'array([-0.69314718, -0.69314718,  0.69314718,  0.69314718])')


def test_distribution_simple():
    data = """
foo,dt
a,2015-1-1
a,2015-1-1
a,2015-1-1
, 2015-1-1
, 2015-1-1
, 2015-1-1
a,2015-2-1
a,2015-2-1
a,2015-2-1
b,2015-2-1
, 2015-2-1
, 2015-2-1
"""

    df = pd.read_csv(StringIO(data), parse_dates=['dt'])
    graphics = distribution(df, 'foo', 'dt', num_buck=10)

    expected_result = """:NdOverlay   [bucket]
   :Spread   [dt]   (objects_rate,obj_rate_l,obj_rate_u)"""

    assert repr(graphics) == expected_result

    assert (repr(graphics.table().data['obj_rate'].values) ==
            'array([ 0.5       ,  0.5       ,  0.        ,  0.16666667,'
            '  0.5       ,\n        0.33333333])')


# pylint: disable=no-member
def test_inter_iso_simple():
    import holoviews as hv
    hv.extension('matplotlib')

    num_obs = 100
    np.random.seed(42)

    predict = np.linspace(0, 1, num_obs)
    target = np.random.binomial(1, predict)
    group = np.repeat(['A'], num_obs)
    dates = np.repeat(pd.datetime(2015, 2, 1), num_obs)

    df = pd.DataFrame({
        'predict': predict,
        'target': target,
        'group': group,
        'dt': dates
    })

    diagram = InterIsoReg(
        df, pdims=['predict'], tdims=['target'],
        ddims=['dt'], gdims=['group']
    )

    assert repr(diagram.isotonic) == ':DynamicMap   [predict,target]'
# pylint: enable=no-member
