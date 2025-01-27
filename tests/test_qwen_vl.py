import unittest

from vlm.qwen_vl import QwenVLEvaluator


class TestQwenVLEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = QwenVLEvaluator(device="cpu")

    def test_load_model(self):
        self.assertIsNotNone(self.evaluator.model)
        self.assertIsNotNone(self.evaluator.processor)

    def test_evaluate(self):
        prompt = "Describe the image."
        image_path = "https://example.com/sample-image.jpg"
        result = self.evaluator.evaluate(prompt, image_path)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
