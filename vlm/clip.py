import torch
from transformers import CLIPProcessor, CLIPModel

from file_io import pil_loader


class CLIPEvaluator:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device="cpu"):
        self.class_names = None  # List of class names to evaluate against
        self.model_id = model_id  # Model identifier for the CLIP model
        self.device = device  # Device to run the model on (CPU or GPU)
        self.model = None  # Placeholder for the CLIP model
        self.processor = None  # Placeholder for the CLIP processor
        self.text_embeddings = None  # Placeholder for text embeddings
        self._load_model()  # Load the model and processor
        # self._calculate_text_embeddings()  # Calculate text embeddings for class names

    def set_class_names(self, class_names):
        self.class_names = class_names
        self._calculate_text_embeddings()

    def _load_model(self):
        # Load the CLIP model and processor from the pretrained model
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def _calculate_text_embeddings(self):
        # Calculate text embeddings for the provided class names
        inputs = self.processor(text=self.class_names, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)

        text_outputs /= text_outputs.norm(dim=-1, keepdim=True)
        self.text_embeddings = text_outputs

    def _load_image(self, image_path):
        # Load and convert the image from the given path
        return pil_loader(image_path)

    def evaluate(self, prompt="", image_path=""):
        # Load the image (from URL or local)
        image = self._load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_outputs = self.model.get_image_features(**inputs)


        image_outputs /= image_outputs.norm(dim=-1, keepdim=True)

        # Calculate similarity between image and text embeddings
        logits_per_image = image_outputs @ self.text_embeddings.T
        probs = logits_per_image.softmax(dim=1)
        return self.class_names[probs.argmax().item()]

if __name__ == "__main__":
    # Usage example
    class_names = ["cat", "dog", "car", "tree"]
    evaluator = CLIPEvaluator(device="cpu")  # Use "cuda" for GPU, or "cpu" for CPU
    evaluator.set_class_names(class_names)
    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg")
    print(result)

    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
    print(result)
