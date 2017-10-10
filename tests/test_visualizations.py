from io import StringIO
import pandas as pd
import numpy as np


from risksutils.visualization import woe_stab, woe_line, InterIsoReg


def test_woe_line_simple_example():
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


def test_woe_stab_simple_example():
    data = """
foo,bar,dt
1,0,2015-1-1
1,1,2015-1-1
1,0,2015-1-1
2,1,2015-1-1
2,0,2015-1-1
2,1,2015-1-1
1,0,2015-2-1
1,1,2015-2-1
1,0,2015-2-1
2,1,2015-2-1
2,0,2015-2-1
2,1,2015-2-1
    """
    df = pd.read_csv(StringIO(data), parse_dates=['dt'])

    graphics = woe_stab(df, 'foo', 'bar', 'dt', num_buck=2)

    expected_result = """:Overlay
   .NdOverlay.I  :NdOverlay   [bucket]
      :Spread   [dt]   (woe,woe_b,woe_u)
   .NdOverlay.II :NdOverlay   [bucket]
      :Curve   [dt]   (woe)"""

    assert repr(graphics) == expected_result

    _, woe_curves = graphics
    assert (repr(woe_curves.table().data['woe'].values) ==
            'array([-0.69314718, -0.69314718,  0.69314718,  0.69314718])')


# pylint: disable=no-member
def test_inter_iso_simple_example():
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
