# from transformers import FlamingoModel, FlamingoProcessor
#
# class FlamingoEvaluator:
#     def __init__(self, model_id="deepmind/flamingo", device="cpu"):
#         self.model_id = model_id
#         self.device = device
#         self.model = None
#         self.processor = None
#         self._load_model()
#
#     def _load_model(self):
#         self.model = FlamingoModel.from_pretrained(self.model_id).to(self.device)
#         self.processor = FlamingoProcessor.from_pretrained(self.model_id)
#
#     def evaluate(self, prompt, image_path):
#         image = self.processor.load_image(image_path)
#         inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(**inputs)
#         return self.processor.decode(outputs[0], skip_special_tokens=True)
