import torch
from transformers import AutoModel, AutoProcessor

from utils import load_and_convert_image


class SigLIPEvaluator:
    def __init__(self, model_id="google/siglip-base-patch16-256-i18n", device="cpu"):
        self.class_names = None  # List of class names to evaluate against
        self.candidate_labels = None  # List of candidate labels for the model
        self.model_id = model_id  # Model identifier for the CLIP model
        self.device = device  # Device to run the model on (CPU or GPU)
        self.model = None  # Placeholder for the CLIP model
        self.processor = None  # Placeholder for the CLIP processor
        self.text_embeddings = None  # Placeholder for text embeddings
        self._load_model()  # Load the model and processor

    def set_class_names(self, class_names):
        self.class_names = class_names
        self.candidate_labels = [
            f"This is a photo of {label}." for label in self.class_names
        ]

    def _load_model(self):
        # Load the CLIP model and processor from the pretrained model
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        # Load and convert the image from the given path
        return load_and_convert_image(image_path)

    def evaluate(self, prompt="", image_path=""):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        inputs = self.processor(
            text=self.class_names,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )
        inputs.to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device):
                outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # these are the probabilities
        # print(f"{probs[0][0]:.1%} that image 0 is '{self.class_names[0]}'")

        return self.class_names[probs.argmax().item()]


if __name__ == "__main__":
    # Usage example
    class_names = ["cat", "dog", "car", "tree"]
    evaluator = SigLIPEvaluator(device="cpu")  # Use "cuda" for GPU, or "cpu" for CPU
    evaluator.set_class_names(class_names)
    result = evaluator.evaluate(
        image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
    )
    print(result)

    result = evaluator.evaluate(
        image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
    )
    print(result)
