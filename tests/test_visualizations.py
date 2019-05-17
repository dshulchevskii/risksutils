# -*- coding: utf-8 -*-

from io import StringIO
import numpy as np
import pandas as pd


from risksutils.visualization import (
    woe_stab, woe_line, InteractiveIsotonic, distribution, isotonic, cross_tab
)


def test_woe_line_simple():

    data = u"""
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
   .Weight_of_evidence.I      :Scatter   [foo]   (woe)
   .Confident_Intervals.I     :ErrorBars   [foo]   (woe,woe_u,woe_b)
   .Logistic_interpolations.I :Curve   [foo]   (logreg)"""

    assert repr(graphics) == expected_result

    scatter, _, _ = graphics
    assert repr(scatter.data.woe.values) == 'array([-0.69314718,  0.69314718])'


def test_woe_stab_simple():
    data = u"""
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
   .Confident_Intervals.I :NdOverlay   [bucket]
      :Spread   [dt]   (woe,woe_b,woe_u)
   .Weight_of_evidence.I  :NdOverlay   [bucket]
      :Curve   [dt]   (woe)"""

    assert repr(graphics) == expected_result

    _, woe_curves = graphics
    assert (repr(woe_curves.table().data['woe'].values) ==
            'array([-0.69314718, -0.69314718,  0.69314718,  0.69314718])')


def test_distribution_simple():
    data = u"""
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

    assert repr(graphics.table().data == """
    bucket         dt  objects_rate  obj_rate_l  obj_rate_u
0        a 2015-01-01      0.500000    0.500000           0
3        a 2015-02-01      0.500000    0.500000           0
1        b 2015-01-01      0.500000    0.000000           0
4        b 2015-02-01      0.666667    0.166667           0
2  missing 2015-01-01      1.000000    0.500000           0
5  missing 2015-02-01      1.000000    0.333333           0
    """)


def test_isotonic_simple():
    num_obs = 100
    np.random.seed(42)

    predict = np.linspace(0, 1, num_obs)
    target = np.random.binomial(1, predict)

    df = pd.DataFrame({
        'predict': predict,
        'target': target
    })

    calibrations = pd.DataFrame({'predict': [0, 1], 'target': [0, 1]})

    graphics = isotonic(df, 'predict', 'target')
    graphics_clbr = isotonic(df, 'predict', 'target', calibrations)

    expected_result_clbr = """:Overlay
   .Isotonic.I              :Curve   [predict]   (isotonic)
   .Confident_Intervals.I   :Area   [predict]   (ci_l,ci_h)
   .Calibration.Calibration :Curve   [predict]   (target)"""

    expected_result = """:Overlay
   .Isotonic.I            :Curve   [predict]   (isotonic)
   .Confident_Intervals.I :Area   [predict]   (ci_l,ci_h)"""

    assert repr(graphics_clbr) == expected_result_clbr
    assert repr(graphics) == expected_result


def test_cross_tab_simple():
    data = u"""
foo,bar,target
1,a,0
2,a,1
3,a,0
4,b,1
5,b,1
,b,0
"""
    df = pd.read_csv(StringIO(data))

    result = cross_tab(df, 'foo', 'bar', 'target', min_sample=0,
                       num_buck1=2, num_buck2=2)
    # can do html representation
    result._repr_html_()  # pylint: disable=protected-access
    rates, counts = result
    expected_rates = ('bar         a     b   All\n'
                      'foo                      \n'
                      '[1; 2]   0.50   NaN  0.50\n'
                      '[3; 5]   0.00  1.00  0.67\n'
                      'missing   NaN  0.00  0.00\n'
                      'All      0.33  0.67  0.50')
    expected_counts = ('bar      a  b  All\n'
                       'foo               \n'
                       '[1; 2]   2  0    2\n'
                       '[3; 5]   1  2    3\n'
                       'missing  0  1    1\n'
                       'All      3  3    6')
    with pd.option_context('display.precision', 2):
        assert repr(rates.data) == expected_rates
        assert repr(counts.data) == expected_counts

    information_val, rates, counts = cross_tab(df, 'foo', 'bar', 'target',
                                               min_sample=0, num_buck1=2,
                                               num_buck2=2, compute_iv=True)
    expected_iv = ('           IV\n'
                   'feature      \n'
                   'foo      2.53\n'
                   'bar      0.46\n'
                   'foo bar  9.19')
    with pd.option_context('display.precision', 2):
        assert repr(rates.data) == expected_rates
        assert repr(counts.data) == expected_counts
        assert repr(information_val) == expected_iv


def test_inter_iso_simple():
    import holoviews as hv
    hv.extension('matplotlib')

    num_obs = 100
    np.random.seed(42)

    df = (
        pd.DataFrame({
            'foo': np.linspace(0, 1, num_obs),
            'group': np.repeat(['A'], num_obs),
            'dt': np.random.choice(pd.date_range('2015-01-01', '2015-01-10'),
                                   num_obs)})
        .assign(bar=lambda x: np.random.binomial(1, x['foo']))
    )

    plot = InteractiveIsotonic(
        df, pdims=['foo'], tdims=['bar'],
        ddims=['dt'], gdims=['group']
    )

    assert (repr(plot.isotonic) ==                  # pylint: disable=no-member
            ':DynamicMap   [predict,target]')

    assert (repr(plot.isotonic.callback('foo',      # pylint: disable=no-member
                                        'bar')) ==
            ':Overlay\n'
            '   .Isotonic.I            :Curve   [predict]   (isotonic)\n'
            '   .Confident_Intervals.I :Area   [predict]   (ci_l,ci_h)')

    assert (repr(plot.group.callback()) ==          # pylint: disable=no-member
            ':Overlay\n'
            '   .Bars.I  :Bars   [group]   (count)\n'
            '   .Bars.II :Bars   [group]   (count)')

    assert (repr(plot.dt.callback()) ==             # pylint: disable=no-member
            ':Overlay\n'
            '   .Area.I  :Area   [dt]   (count)\n'
            '   .Area.II :Area   [dt]   (count)')
