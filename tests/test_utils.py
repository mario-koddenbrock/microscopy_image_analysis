import unittest
from PIL import Image
from utils import load_and_convert_image

class TestUtils(unittest.TestCase):
    def test_load_and_convert_image_from_url(self):
        image_path = "https://www.htw-berlin.de/files/Presse/_tmp_/d/a/csm_Startseite_UmweltKlimaschutz_Kachel_6c93892556.jpg"
        image = load_and_convert_image(image_path)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_load_and_convert_image_from_local(self):
        image_path = "datasets/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"
        image = load_and_convert_image(image_path)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_load_and_convert_image_invalid_path(self):
        with self.assertRaises(ValueError):
            load_and_convert_image("invalid/path")

if __name__ == "__main__":
    unittest.main()