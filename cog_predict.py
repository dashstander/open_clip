# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
import torch
import sys
# Working directory for cog docker image is /src --> `open_clip` module is in /src/src
# Figure out if there's a better way to do this
sys.path.append('/src/src')
from open_clip import create_model_and_transforms, tokenizer

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = 'cuda:0'
        self.model, _, self.preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
        self.model.eval().to(self.device)

    def predict(
        self,
        caption: str = Input(description='The image caption'),
        input_image: Path = Input(description='RGB input image'),
    ) -> float:
        """Run a single prediction on the model"""
        image_tensor = self.preprocess(Image.open(input_image).convert('RGB')).unsqueeze(0).to(self.device)
        text_tokens = tokenizer.tokenize([caption]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy() @ image_features.cpu().numpy().T
