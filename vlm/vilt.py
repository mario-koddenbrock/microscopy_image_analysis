from transformers import ViltModel, ViltProcessor

from mia.file_io import pil_loader


class ViltEvaluator:
    def __init__(self, model_id="dandelin/vilt-b32-mlm", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        self.model = ViltModel.from_pretrained(self.model_id).to(self.device)
        self.processor = ViltProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
