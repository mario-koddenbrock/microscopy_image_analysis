from transformers import Blip2ForConditionalGeneration, Blip2Processor
from mia.file_io import pil_loader


class BLIP2Evaluator:
    def __init__(self, model_id="Salesforce/blip2-opt-2.7b", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_id).to(
            self.device
        )
        self.processor = Blip2Processor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path, max_new_tokens=50):
        # Load the image (from URL or local)
        image = self._load_image(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        print(f"BLIP-2: {generated_text}")
        return generated_text


if __name__ == "__main__":
    # Usage example
    evaluator = BLIP2Evaluator(device="cuda")  # Use "cuda" for GPU, or "cpu" for CPU
    result = evaluator.evaluate(
        prompt="Describe the image.", image_path="https://example.com/sample-image.jpg"
    )
    print(result)
