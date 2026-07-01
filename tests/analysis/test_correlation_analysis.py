import os
import unittest
import pandas as pd
import numpy as np

from data_juicer.analysis.correlation_analysis import is_numeric_list_series, CorrelationAnalysis
from data_juicer.utils.constant import Fields

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CorrelationAnalysisTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [[1, 2, 3], [4, 5], [6]],
            'C': ['a', 'b', 'c'],
            'D': [[1, 'a'], [2, 3], [4]],
            'E': [[], [7, 8, 9], [10.5]],
            'F': [True, False, True],
            'G': [[1, 2], None, [3, 4]],
            'H': [1.1, 2.2, 3.3],
            'I': [[1, 2], [3, 4], [5, 6]],
            'J': [np.nan, np.nan, np.nan],
            'K': [[], [], []],
        })
        self.temp_output_path = 'tmp/test_correlation_analysis/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

        super().tearDown()

    def test_is_numeric_list_series(self):
        res = {
            'A': False,
            'B': True,
            'C': False,
            'D': False,
            'E': True,
            'F': False,
            'G': True,
            'H': False,
            'I': True,
            'J': False,
            'K': False,
        }
        temp = self.df.copy()
        for col in temp.columns:
            self.assertEqual(is_numeric_list_series(temp[col]), res[col])

    def test_correlation_analysis(self):
        corr_analyzer = CorrelationAnalysis({Fields.stats: self.df}, self.temp_output_path)
        self.assertEqual(set(corr_analyzer.stats.columns), {'A', 'B', 'E', 'G', 'H', 'I', 'J'})

        ret = corr_analyzer.analyze(skip_export=True)
        self.assertFalse(os.path.exists(os.path.join(self.temp_output_path, 'stats-corr-pearson.png')))
        self.assertIsInstance(ret, pd.DataFrame)

        ret = corr_analyzer.analyze()
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'stats-corr-pearson.png')))
        self.assertIsInstance(ret, pd.DataFrame)

        with self.assertRaises(AssertionError):
            _ = corr_analyzer.analyze(method='unknown_method')

        corr_analyzer = CorrelationAnalysis({Fields.stats: {}}, self.temp_output_path)
        ret = corr_analyzer.analyze()
        self.assertIsNone(ret)

    def test_init_drops_string_dtype_columns(self):
        """回归:dataset[Fields.stats] 含有 pandas StringDtype 列时,
        CorrelationAnalysis.__init__ 不应再抛
        ``TypeError: Cannot interpret '<StringDtype(na_value=nan)>'
        as a data type``(由 numpy np.issubdtype 触发)。

        字符串列应在 init 阶段被 drop,后续 analyze() 走通。
        """
        # 模拟生产路径:从 jsonl 读出来的字符串列就是 StringDtype(na_value=nan)
        # 这种 dtype 在 numpy 眼里不是合法 dtype,np.issubdtype 直接抛 TypeError
        string_col = pd.array(
            ['你好', '世界', None], dtype=pd.StringDtype(na_value=np.nan))
        df = pd.DataFrame({
            'answer': string_col,
            'category':
                pd.array(['A', 'B', 'C'], dtype='string'),
            'score': [1.0, 2.0, 3.0],  # 标量数值列保留
        })
        ds = {Fields.stats: df}

        # 不应抛 TypeError(未 patch 前 __init__ 在 np.issubdtype 处崩)
        corr_analyzer = CorrelationAnalysis(ds, self.temp_output_path)
        # 字符串列被 drop,只剩 score
        self.assertNotIn('answer', corr_analyzer.stats.columns)
        self.assertNotIn('category', corr_analyzer.stats.columns)
        self.assertIn('score', corr_analyzer.stats.columns)
        # analyze 走通(只有 1 列不画 heatmap,返回 None 也 OK)
        corr_analyzer.analyze()

    def test_issubdtype_stringdtype_raises(self):
        """直接验证 np.issubdtype 在生产环境会抛 TypeError,确保我们
        的修复有意义(否则说明这条路径根本不会触发 bug)。"""
        string_col = pd.array(
            ['x', 'y', None], dtype=pd.StringDtype(na_value=np.nan))
        s = pd.Series(string_col)
        # 触发原 bug 的最小调用
        with self.assertRaises(TypeError):
            np.issubdtype(s.dtype, np.number)


if __name__ == '__main__':
    unittest.main()
