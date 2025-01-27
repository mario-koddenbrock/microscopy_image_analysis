import unittest

from vlm.phi_vision import PhiVisionEvaluator


class TestPhiVisionEvaluator(unittest.TestCase):
    def setUp(self):
        # Set up any state specific to the test case.
        self.evaluator = PhiVisionEvaluator(device="cpu")

    def test_load_model(self):
        # Test if the model and processor are loaded correctly.
        self.assertIsNotNone(self.evaluator.model)
        self.assertIsNotNone(self.evaluator.processor)

    def test_evaluate(self):
        # Test the evaluate method with a sample prompt and image URL.
        prompt = "Describe the image."
        image_path = "https://example.com/sample-image.jpg"
        result = self.evaluator.evaluate(prompt, image_path)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
