from transformers import VisualBertModel, BertTokenizer, ViltProcessor
from PIL import Image

from file_io import pil_loader


class VisualBertEvaluator:
    def __init__(self, model_id="uclanlp/visualbert-vqa", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()


    def set_class_names(self, class_names):
        self.class_names = class_names

    def _load_model(self):
        self.model = VisualBertModel.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        # Truncate the prompt to the maximum sequence length
        max_length = self.processor.tokenizer.model_max_length
        truncated_prompt = self.processor.tokenizer.encode(prompt, max_length=max_length, truncation=True)

        # Process the inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Ensure the correct input keys are passed to the model
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                             token_type_ids=inputs['token_type_ids'])

        return outputs