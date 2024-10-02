import unittest

from mia import file_io
from vlm.pali_gemma import PaliGemmaEvaluator
from vlm.phi_vision import PhiVisionEvaluator
from vlm.qwen_vl import QwenVLEvaluator


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.evaluators = {
            "PhiVision": PhiVisionEvaluator(device="cpu"),
            "PaliGemma": PaliGemmaEvaluator(device="cpu"),
            "QwenVL": QwenVLEvaluator(device="cpu"),
        }
        self.image_path = "https://www.htw-berlin.de/files/Presse/_tmp_/d/a/csm_Startseite_UmweltKlimaschutz_Kachel_6c93892556.jpg"
        self.prompt = "Describe the image."

    def test_evaluators(self):
        image = file_io.rasterio_loader(self.image_path)
        for name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(self.prompt, image)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

if __name__ == "__main__":
    unittest.main()