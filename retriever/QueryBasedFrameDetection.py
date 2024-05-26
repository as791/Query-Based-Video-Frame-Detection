from io import BytesIO
from django.http import JsonResponse
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
import torch
import faiss
import boto3
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
import pickle
import falcon
import base64

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ClipBasedFrameRetriever:
    def __init__(self):
        # AWS credentials
        self.aws_access_key_id = 'AKIA2SQC4BMAYRGKLZDH'
        self.aws_secret_access_key = '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'
        self.aws_region = 'us-east-1'

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
        
        _, indices = self.index.search(query_embedding, top_k)
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
    
    def download_video_from_s3(self, s3_bucket, s3_key):
        s3 = boto3.client('s3',aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region)
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        video_data = response['Body'].read()
        video_path = '/tmp/temp_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_data)
        return video_path

    def extract_frames_from_video(self, video_path):
        clip = VideoFileClip(video_path)
        frames = []
        for frame in clip.iter_frames():
            img = Image.fromarray(frame)
            frames.append(img)
        return frames
    
    def encode_image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

class ProcessFramesResource:
    def on_post(self, req, resp):
        data = req.media
        s3_path = data.get('s3_path')
        context = data.get('context')
        if not s3_path or not context:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "s3_path and context are required"}
            return

        retriever = ClipBasedFrameRetriever()
        s3_bucket = s3_path.split('/')[2]
        s3_key = '/'.join(s3_path.split('/')[3:])

        # Download and process the video
        video_path = retriever.download_video_from_s3(s3_bucket, s3_key)
        images = retriever.extract_frames_from_video(video_path)

        relevant_texts = retriever.retrieve_relevant_texts(context)
        most_similar_classes = retriever.retrieve_most_similar_class(relevant_texts, images)

        # Encode images to base64
        encoded_images = [retriever.encode_image_to_base64(img) for img in images]

        result = [{"frame": frame, "most_similar_class": cls} for frame, cls in zip(encoded_images, most_similar_classes)]
        resp.media = result