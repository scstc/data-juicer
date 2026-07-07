import os
import os
import tempfile
import unittest
import pandas as pd

from data_juicer.core.data import NestedDataset
from data_juicer.analysis.column_wise_analysis import get_row_col, ColumnWiseAnalysis
from data_juicer.utils.constant import DEFAULT_PREFIX, Fields

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class GetRowColFuncTest(DataJuicerTestCaseBase):

    def _run_test_data(self, data):
        for args, truth in data.items():
            res = get_row_col(*args)
            self.assertEqual(res, truth)

    def test_normal_func(self):
        test_data = {
            (4, 2): (2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]),
            (3, 3): (3, 1, [(0, 0), (1, 0), (2, 0)]),
        }
        self._run_test_data(test_data)

    def test_marginal_total_num(self):
        test_data = {
            (1, 1): (1, 1, [(0, 0)]),
            (0, 1): (0, 0, []),
            (-1, 1): (0, 0, []),
        }
        self._run_test_data(test_data)

    def test_marginal_factor(self):
        test_data = {
            (4, 0): (0, 0, []),
            (4, -1): (0, 0, []),
            (4, 1): (2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]),
        }
        self._run_test_data(test_data)


class ColumnWiseAnalysisTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        data_list = [
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}meta_str1': 'human',
                    f'{DEFAULT_PREFIX}meta_str2': 'sft',
                    'meta_str3': 'code',
                },
                Fields.stats: {
                    'stats_num_list': [4, 5, 6],
                    'stats_num': 3.1,
                    'stats_str': 'zh',
                }
            },
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}meta_str1': 'assistant',
                    f'{DEFAULT_PREFIX}meta_str2': 'rlhf',
                    'meta_str3': 'math',
                },
                Fields.stats: {
                    'stats_num_list': [7, 8, 9],
                    'stats_num': 4.1,
                    'stats_str': 'en',
                }
            },
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}meta_str1': 'system',
                    f'{DEFAULT_PREFIX}meta_str2': 'dpo',
                    'meta_str3': 'reasoning',
                },
                Fields.stats: {
                    'stats_num_list': [10, 11, 12],
                    'stats_num': 5.1,
                    'stats_str': 'fr',
                }
            },
        ]
        self.dataset_3_sample = NestedDataset.from_list(data_list)

        data_list.append({
            Fields.meta: {
                f'{DEFAULT_PREFIX}meta_str1': 'robot',
                f'{DEFAULT_PREFIX}meta_str2': 'sft',
                'meta_str3': 'edu',
            },
            Fields.stats: {
                'stats_num_list': [13, 14, 15],
                'stats_num': 2.1,
                'stats_str': 'it',
            }
        })
        self.dataset_4_sample = NestedDataset.from_list(data_list)
        self.temp_output_path = 'tmp/test_column_wise_analysis/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

        super().tearDown()

    def test_init(self):
        column_wise_analysis = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path)
        # test if the non-tag columns are removed
        self.assertNotIn('meta_str3', column_wise_analysis.meta.columns)
        self.assertIn(f'{DEFAULT_PREFIX}meta_str1',
                      column_wise_analysis.meta.columns)
        self.assertIn(f'{DEFAULT_PREFIX}meta_str2',
                      column_wise_analysis.meta.columns)
        # test if overall_result is None
        self.assertIsInstance(column_wise_analysis.overall_result, pd.DataFrame)
        self.assertEqual(column_wise_analysis.save_stats_in_one_file, True)

        # test for specify overall_result
        column_wise_analysis = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path, overall_result='temp_palceholder')
        self.assertEqual(column_wise_analysis.overall_result, 'temp_palceholder')
        # test for save_stats_in_one_file is False
        column_wise_analysis = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path, save_stats_in_one_file=False)
        self.assertEqual(column_wise_analysis.save_stats_in_one_file, False)

        # test number of stats and meta
        self.assertEqual(len(column_wise_analysis.stats), 3)
        self.assertEqual(len(column_wise_analysis.meta), 3)
        self.assertEqual(len(column_wise_analysis.stats.columns), 3)
        self.assertEqual(len(column_wise_analysis.meta.columns), 2)

    def test_basic_analyze(self):
        # test basic analyze
        column_wise_analysis_3_sample = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path)
        column_wise_analysis_3_sample.analyze()
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'all-stats.png')))

    def test_skip_export(self):
        # test skip_export
        column_wise_analysis_4_sample = ColumnWiseAnalysis(
            self.dataset_4_sample, self.temp_output_path)
        column_wise_analysis_4_sample.analyze(skip_export=True)
        for stats in column_wise_analysis_4_sample.stats.columns:
            self.assertFalse(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-hist.png')))
            self.assertFalse(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-box.png')))
        for meta in column_wise_analysis_4_sample.meta.columns:
            self.assertFalse(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-hist.png')))
            self.assertFalse(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-wordcloud.png')))

    def test_not_save_stats_in_one_file(self):
        # test analyze with save_stats_in_one_file is False
        column_wise_analysis_3_sample = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path,
            save_stats_in_one_file=False)
        column_wise_analysis_3_sample.analyze()
        for stats in column_wise_analysis_3_sample.stats.columns:
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-hist.png')))
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-box.png'))
                            or os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-wordcloud.png')))
        for meta in column_wise_analysis_3_sample.meta.columns:
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-hist.png')))
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-wordcloud.png')))

    def test_save_stats_in_one_file(self):
        # test analyze with percentiles is True
        column_wise_analysis_3_sample = ColumnWiseAnalysis(
            self.dataset_3_sample, self.temp_output_path,
            save_stats_in_one_file=False)
        column_wise_analysis_3_sample.analyze(show_percentiles=True)
        for stats in column_wise_analysis_3_sample.stats.columns:
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-hist.png')))
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-box.png'))
                            or os.path.exists(
                os.path.join(self.temp_output_path, f'{stats}-wordcloud.png')))
        for meta in column_wise_analysis_3_sample.meta.columns:
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-hist.png')))
            self.assertTrue(os.path.exists(
                os.path.join(self.temp_output_path, f'{meta}-wordcloud.png')))

    def test_wordcloud_with_multiline_text(self):
        """回归:样本字段值含 \\n / \\r / \\t 时,WordCloud 不应触发
        Pillow ImageText 多行 anchor 限制 (ValueError)。"""

        # 构造会触发原 bug 的样本:一个 meta 字符串含 \n + \t
        data_list = [
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}m_str':
                        'line1\nline2\tline3',  # ← 含 \n \t
                },
                Fields.stats: {'stats_num': 1.0},
            },
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}m_str':
                        'line1\nline2\tline3',  # ← 重复,应被合并到同一个 key
                },
                Fields.stats: {'stats_num': 2.0},
            },
            {
                Fields.meta: {
                    f'{DEFAULT_PREFIX}m_str': 'plain',  # ← 干净样本
                },
                Fields.stats: {'stats_num': 3.0},
            },
        ]
        ds = NestedDataset.from_list(data_list)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwa = ColumnWiseAnalysis(ds, tmpdir, save_stats_in_one_file=False)
            # 跑全流程;若 _flatten_multiline 没生效,这里会因 wordcloud→PIL
            # 报 ValueError("anchor not supported for multiline text") 退出
            cwa.analyze()
            wc_path = os.path.join(tmpdir, f'{DEFAULT_PREFIX}m_str-wordcloud.png')
            self.assertTrue(os.path.exists(wc_path),
                            f'wordcloud png should be generated at {wc_path}')
            # 关键校验:全流程跑通且 PNG 落盘即视为通过
            self.assertGreater(os.path.getsize(wc_path), 0)

    def test_wordcloud_flattens_keys(self):
        """单元验证 draw_wordcloud 内 _flatten_multiline 把含 \\n 的字符串折叠成单空格。"""
        data_list = [{
            Fields.meta: {f'{DEFAULT_PREFIX}m': 'alpha\nbeta'},
            Fields.stats: {'s': 1.0},
        }, {
            Fields.meta: {f'{DEFAULT_PREFIX}m': 'alpha\nbeta'},
            Fields.stats: {'s': 2.0},
        }, {
            Fields.meta: {f'{DEFAULT_PREFIX}m': 'gamma'},
            Fields.stats: {'s': 3.0},
        }]
        ds = NestedDataset.from_list(data_list)
        with tempfile.TemporaryDirectory() as tmpdir:
            cwa = ColumnWiseAnalysis(ds, tmpdir, save_stats_in_one_file=False)
            out = os.path.join(tmpdir, 'wc.png')
            cwa.draw_wordcloud(None, cwa.meta.iloc[:, 0], out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == '__main__':
    unittest.main()
