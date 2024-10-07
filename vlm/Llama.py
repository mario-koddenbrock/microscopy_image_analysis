from transformers import AutoProcessor, AutoModelForCausalLM

from utils import load_and_convert_image


class LlamaEvaluator:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        # Load the model and processor, and move the model to the appropriate device (CPU or GPU)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)

        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        return generated_text


if __name__ == "__main__":
    # Usage:
    evaluator = LlamaEvaluator()
    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg")
    print(result)

    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
    print(result)
