from transformers import AutoModelForCausalLM, AutoTokenizer

from mia.file_io import pil_loader


class QwenVLEvaluator:
    def __init__(self, model_id="qwen/Qwen-VL", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        # Load the tokenizer and model with the specified settings, moving to the appropriate device (CPU/GPU)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",  # Device map can be customized
            trust_remote_code=True,
        ).to(self.device)

    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        # Prepare the input in list format (with image and text prompt)
        query = self.tokenizer.from_list_format(
            [
                {"image": image_path},
                {"text": prompt},
            ]
        )

        # Convert query to tensor inputs for the model
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

        # Generate the response
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

        print(f"Qwen-VL: {response}")
        return response


if __name__ == "__main__":
    # Usage example
    evaluator = QwenVLEvaluator(device="cuda")  # Use "cuda" for GPU, or "cpu" for CPU
    result = evaluator.evaluate(
        prompt="Describe the image.", image_path="https://example.com/sample-image.jpg"
    )
    print(result)
