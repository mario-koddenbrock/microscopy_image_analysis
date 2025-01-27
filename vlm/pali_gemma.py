import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from mia.file_io import pil_loader


# general model: google/paligemma-3b-mix-224

# Model Performance Table on Fine-tuned Checkpoints
# Model Name	Dataset/Task	Score in Transferred Task
# paligemma-3b-ft-vqav2-448	Diagram Understanding	85.64 Accuracy on VQAV2
# paligemma-3b-ft-cococap-448	COCO Captions	144.6 CIDEr
# paligemma-3b-ft-science-qa-448	Science Question Answering	95.93 Accuracy on ScienceQA Img subset with no CoT
# paligemma-3b-ft-refcoco-seg-896	Understanding References to Specific Objects in Images	76.94 Mean IoU on refcoco
# paligemma-3b-ft-rsvqa-hr-224	Remote Sensing Visual Question Answering	92.61 Accuracy on test


class PaliGemmaEvaluator:
    def __init__(self, model_id="google/paligemma-3b-ft-cococap-448", device="cpu"):
        self.class_names = None
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def set_class_names(self, class_names):
        self.class_names = class_names

    def _load_model(self):
        # Load the model and processor, and move the model to the appropriate device (CPU or GPU)
        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(self.model_id)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return pil_loader(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        # Prepare inputs for the model
        model_inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        # Generate the output
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        # check which of the class names is contained in the decoded text and return it
        for class_name in self.class_names:
            # remove underscores and lowercase the class name
            class_name = class_name.replace("_", " ").lower()
            if class_name.lower() in decoded.lower():
                return class_name

        decoded = decoded.replace("The image is a", "").strip()

        return decoded


if __name__ == "__main__":
    # Usage:
    evaluator = PaliGemmaEvaluator()
    result = evaluator.evaluate(
        image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
    )
    print(result)

    result = evaluator.evaluate(
        image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
    )
    print(result)
