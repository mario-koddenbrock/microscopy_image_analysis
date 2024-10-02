import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from mia.file_io import pil_loader


class PaliGemmaEvaluator:
    def __init__(self, model_id="google/paligemma-3b-mix-224", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        # Load the model and processor, and move the model to the appropriate device (CPU or GPU)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        # Prepare inputs for the model
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        # Generate the output
        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded


if __name__ == "__main__":
    # Usage:
    evaluator = PaliGemmaEvaluator()
    result = evaluator.evaluate(prompt="Your prompt here", image_path="image_url_here")
