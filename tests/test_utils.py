import unittest
from unittest.mock import patch, MagicMock

from mia.utils import (
    initialize_evaluators,
    evaluate_image,
    evaluate_dataset,
    get_class_files,
    get_dataset_classes,
)


class TestUtils(unittest.TestCase):

    @patch("mia.utils.time.time", return_value=1234567890)
    def test_initialize_evaluators(self, mock_time):
        config = {"evaluator1": MagicMock(), "evaluator2": MagicMock()}
        device = "cpu"
        evaluators = initialize_evaluators(device, config)
        self.assertIn("evaluator1", evaluators)
        self.assertIn("evaluator2", evaluators)
        self.assertTrue(mock_time.called)

    @patch(
        "mia.utils.plotting.show_prediction_result", return_value="result_image_path"
    )
    def test_evaluate_image(self, mock_show_prediction_result):
        evaluators = {"evaluator1": MagicMock()}
        image_path = "path/to/image.jpg"
        result = evaluate_image(evaluators, image_path)
        self.assertEqual(result, "result_image_path")
        self.assertTrue(mock_show_prediction_result.called)

    @patch("mia.utils.get_dataset_classes", return_value=["class1", "class2"])
    @patch(
        "mia.utils.get_class_files",
        return_value=[
            "datasets/Classification/Acinetobacter.baumanii/Acinetobacter.baumanii_0001.tif",
            "datasets/Classification/Acinetobacter.baumanii/Acinetobacter.baumanii_0002.tif",
        ],
    )
    @patch(
        "mia.utils.plotting.show_prediction_result", return_value="result_image_path"
    )
    def test_evaluate_dataset(
        self,
        mock_show_prediction_result,
        mock_get_class_files,
        mock_get_dataset_classes,
    ):
        evaluators = {"evaluator1": MagicMock()}
        dataset_name = "dataset"
        dataset_description = "description"
        dataset_path = "path/to/dataset"
        num_images_per_class = 1
        num_classes = 2
        result = evaluate_dataset(
            evaluators,
            dataset_name,
            dataset_description,
            dataset_path,
            num_images_per_class,
            num_classes,
        )
        self.assertEqual(result, ["result_image_path", "result_image_path"])
        self.assertTrue(mock_show_prediction_result.called)

    @patch("os.listdir", return_value=["image1.jpg", "image2.jpg"])
    @patch("os.path.exists", return_value=True)
    def test_get_class_files(self, mock_exists, mock_listdir):
        dataset_path = "path/to/dataset"
        class_name = "class1"
        result = get_class_files(dataset_path, class_name)
        self.assertEqual(
            result,
            ["path/to/dataset/class1/image1.jpg", "path/to/dataset/class1/image2.jpg"],
        )
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_listdir.called)

    @patch("os.listdir", return_value=["class1", "class2"])
    def test_get_dataset_classes(self, mock_listdir):
        dataset_path = "path/to/dataset"
        result = get_dataset_classes(dataset_path)
        self.assertEqual(result, ["class1", "class2"])
        self.assertTrue(mock_listdir.called)


if __name__ == "__main__":
    unittest.main()
