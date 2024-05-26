from io import BytesIO
from django.http import JsonResponse
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
import torch
from datasets import load_dataset
from rest_framework.views import APIView
import os
import faiss
import boto3
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import torch.nn.functional as F
import pickle

class ClipBasedFrameRetriever:
    def __init__(self):
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Load BERT model and tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # Load a large text corpus
        self.index = faiss.read_index("faiss_index.bin")
        with open("corpus.pkl", "rb") as f:
            self.corpus = pickle.load(f)

    def clip_image_encoder(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs

    def clip_text_encoder(self, text_corpus):
        inputs = self.clip_processor(text=text_corpus, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_text_features(**inputs)
        return outputs
    
    def retrieve_relevant_texts(self, query, top_k=5):
        inputs = self.bert_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        query_embedding = self.bert_model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_texts = [self.corpus[i] for i in indices[0]]
        
        return relevant_texts

    def retrieve_most_similar_class(self, text_corpus, images):
        with torch.no_grad():
            text_embeddings = F.normalize(self.clip_text_encoder(text_corpus), p=2, dim=-1)
            image_embeddings = F.normalize(self.clip_image_encoder(images), p=2, dim=-1)
            
            most_similar_classes = []
            for image_embedding in image_embeddings:
                scores = []
                for text_embedding in text_embeddings:
                    score = torch.cosine_similarity(text_embedding, image_embedding, dim=0)
                    scores.append(score.item())
                most_similar_class_index = np.argmax(scores)
                most_similar_classes.append(text_corpus[most_similar_class_index])
        return most_similar_classes

    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image
    
    def get_image_from_s3(self, s3_bucket, s3_key):
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image


class ProcessFramesView(APIView):
    def post(self, request):
        s3_path = request.data.get('s3_path')
        context = request.data.get('context')
        if not s3_path or not context:
            return JsonResponse({"error": "s3_path and context are required"}, status=400)

        retriever = ClipBasedFrameRetriever()
        s3_bucket = s3_path.split('/')[2]
        s3_prefix = '/'.join(s3_path.split('/')[3:])

        # List objects in the S3 path
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
        frames = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.jpeg')]

        images = [retriever.get_image_from_s3(s3_bucket, frame) for frame in frames]
        relevant_texts = retriever.retrieve_relevant_texts(context)
        most_similar_classes = retriever.retrieve_most_similar_class(relevant_texts, images)

        result = [{"frame": frame, "most_similar_class": cls} for frame, cls in zip(frames, most_similar_classes)]
        return JsonResponse(result, safe=False)