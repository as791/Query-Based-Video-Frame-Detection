from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import torch.nn.functional as F

class ClipBasedFrameRetriever:
    def __init__(self):
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def clip_image_encoder(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs

    def clip_text_encoder(self, text_corpus):
        inputs = self.clip_processor(text=text_corpus, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_text_features(**inputs)
        return outputs

    def retrieve_relevant_frames(self, text_corpus, images):
        # Tokenize the text prompt
        with torch.no_grad():
            text_embeddings = F.normalize(self.clip_text_encoder(text_corpus), p=2, dim=-1)
            image_embeddings = F.normalize(self.clip_image_encoder(images), p=2, dim=-1)
            
            matrix = []
            for image_embedding in image_embeddings:
                scores = []
                for text_embedding in text_embeddings:
                    score = torch.cosine_similarity(text_embedding, image_embedding, dim=0)
                    scores.append(score.item())
                matrix.append(scores)
        return matrix

    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

def main():
    retriever = ClipBasedFrameRetriever()
    
    video_frames = sorted(glob.glob('/Users/aryaman.sinha/frames/*.jpeg'))
    images = [retriever.get_image(image_path) for image_path in video_frames]
    corpus = ["man is running", "cat is sitting", "A photo of dog", "A photo of cat"]
    
    matrix = retriever.retrieve_relevant_frames(corpus, images)

    print(matrix)

if __name__ == "__main__":
    main()
