import unittest
import doctest
import numpy as np
import pandas as pd

from risksutils import woe


class WoeLineTests(unittest.TestCase):

    def setUp(self):
        import holoviews as hv
        hv.extension('matplotlib')

    def test_simple_example(self):
        feature = np.array([1, 1, 1, 2, 2, 2])
        target = np.array([0, 1, 0, 1, 0, 1])
        df = pd.DataFrame({'foo': feature,
                           'bar': target})
        graphics = woe.woe_line(df, 'foo', 'bar', num_buck=2)
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
        s1, _, _ = woe.woe_line(df, 'foo', 'bar', num_buck=2)
        s2, _, _ = woe.woe_line(df.dropna(), 'foo', 'bar', num_buck=2)
        self.assertTrue(all(s1.data == s2.data))


if __name__ == '__main__':
    unittest.main()
    doctest.testmod(woe, verbose=True)
