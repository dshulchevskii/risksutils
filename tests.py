import unittest
import doctest
import numpy as np
import pandas as pd

from risksutils.visualization import woe_line, woe_stab


class WoeLineTests(unittest.TestCase):

    def setUp(self):
        import holoviews as hv
        hv.extension('matplotlib')

    def test_simple_example(self):
        feature = np.array([1, 1, 1, 2, 2, 2])
        target = np.array([0, 1, 0, 1, 0, 1])
        df = pd.DataFrame({'foo': feature,
                           'bar': target})
        graphics = woe_line(df, 'foo', 'bar', num_buck=2)
        expected_result = (
            ':Overlay\n'
            '   .Weight_of_evidence.Foo      :Scatter   [foo]   (woe)\n'
            '   .Confident_Intervals.Foo     :ErrorBars   [foo]   (woe,woe_u,woe_b)\n'
            '   .Logistic_interpolations.Foo :Curve   [foo]   (logreg)'
        )
        self.assertEqual(repr(graphics), expected_result)
        scatter, _, _ = graphics
        self.assertEqual(repr(scatter.data.woe.values),
                          'array([-0.69314718,  0.69314718])')

    def test_missing_values(self):
        feature = np.array([1, 1, 1, 2, 2, 2, float('nan'), 1])
        target = np.array([0, 1, 0, 1, 0, 1, 1, float('nan')])
        df = pd.DataFrame({'foo': feature,
                           'bar': target})
        s1, _, _ = woe_line(df, 'foo', 'bar', num_buck=2)
        s2, _, _ = woe_line(df.dropna(), 'foo', 'bar', num_buck=2)
        self.assertTrue(all(s1.data == s2.data))

class WoeStabTests(unittest.TestCase):

    def test_simple_plot(self):
        df = pd.DataFrame({
            'foo': [1, 1, 1, 2, 2, 2] * 2,
            'bar': [0, 1, 0, 1, 0, 1] * 2,
            'dt' : (  [pd.datetime(2015, 1, 1)] * 6
                    + [pd.datetime(2015, 2, 1)] * 6)
        })
        graphics = woe_stab(df, 'foo', 'bar', 'dt', 2)
        expected_result = (
            ':Overlay\n'
            '   .NdOverlay.I  :NdOverlay   [bucket]\n'
            '      :Spread   [dt]   (woe,woe_b,woe_u)\n'
            '   .NdOverlay.II :NdOverlay   [bucket]\n'
            '      :Curve   [dt]   (woe)'
        )
        self.assertTrue(repr(graphics), expected_result)
        ci, woe_curves = graphics
        self.assertEqual(
            repr(woe_curves.table().data['woe'].values),
            'array([-0.69314718, -0.69314718,  0.69314718,  0.69314718])')


if __name__ == '__main__':
    unittest.main()
    import risksutils.visualization.woe
    doctest.testmod(risksutils.visualization.woe, verbose=True)
