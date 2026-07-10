import ast
import pathlib
import unittest

from tqdm import tqdm


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProgressDisplayTests(unittest.TestCase):
    def test_short_description_keeps_percentage_and_frame_count_in_narrow_terminal(self):
        for description in ('处理视频', '分析视频'):
            with self.subTest(description=description):
                rendered = tqdm.format_meter(
                    27578,
                    55156,
                    1200,
                    prefix=description,
                    ncols=30,
                )

                self.assertIn('50%', rendered)
                self.assertIn('27578/55156', rendered)

    def test_video_progress_bars_keep_progress_visible_for_long_file_names(self):
        expected_descriptions = {
            'main.py': ['处理视频'],
            'main_nipple.py': ['处理视频'],
            'main_nipple_rog.py': ['处理视频'],
            'main_yolov5.py': ['处理视频', '分析视频'],
        }

        for file_name, descriptions in expected_descriptions.items():
            with self.subTest(file_name=file_name):
                source = (ROOT / file_name).read_text(encoding='utf-8')
                tree = ast.parse(source, filename=file_name)
                self.assertIn('开始处理视频文件: {video_path}', source)
                tqdm_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'tqdm'
                ]

                self.assertEqual(len(tqdm_calls), len(descriptions))
                actual_descriptions = []
                for call in tqdm_calls:
                    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                    desc = keywords.get('desc')
                    dynamic_ncols = keywords.get('dynamic_ncols')

                    self.assertIsInstance(desc, ast.Constant)
                    actual_descriptions.append(desc.value)
                    self.assertIsInstance(dynamic_ncols, ast.Constant)
                    self.assertIs(dynamic_ncols.value, True)

                self.assertCountEqual(actual_descriptions, descriptions)


if __name__ == '__main__':
    unittest.main()
