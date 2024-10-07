import base64
import os

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

class PixtralVisionEvaluator:
    def __init__(self, model_id="pixtral-12b-2409", device="cpu"):
        api_key = os.getenv("MISTRAL_API_KEY")
        self.model = model_id
        self.client = Mistral(api_key=api_key)

    def _load_image(self, image_path):

        if image_path.startswith('http://') or image_path.startswith('https://'):
            return image_path
        elif os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_image}"
        else:
            raise ValueError(f"Invalid image path: {image_path}")




    def evaluate(self, prompt, image_path):
        # # Load the image (from URL or local path)
        image = self._load_image(image_path)

        chat_response = self.client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": image
                        }
                    ]
                },
            ]
        )
        return chat_response.choices[0].message.content

if __name__ == "__main__":
    # Usage example
    evaluator = PixtralVisionEvaluator(device="cuda")  # Use "cuda" for GPU acceleration
    result = evaluator.evaluate(
        prompt="Describe the image in detail.",
        image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
    )
    print(result)
