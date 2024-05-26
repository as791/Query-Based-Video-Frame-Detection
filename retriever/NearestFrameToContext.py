import os
import uuid
import boto3
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import falcon
from moviepy.editor import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertModel

# Set the environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FrameRetriever:
    def __init__(self):
         # AWS credentials
        self.aws_access_key_id = 'AKIA2SQC4BMAYRGKLZDH'
        self.aws_secret_access_key = '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'
        self.aws_region = 'us-east-1'

        
        # S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
        
        # Load BLIP model and processor
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Load BERT model and tokenizer for similarity calculation
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def generate_captions(self, images):
        captions = []
        for image in images:
            inputs = self.blip_processor(images=image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        return captions

    def calculate_similarity(self, context, frame_captions):
        context_embedding = self.embed_text(context)
        caption_embeddings = self.embed_texts(frame_captions)

        similarities = []
        for caption_embedding in caption_embeddings:
            score = torch.nn.functional.cosine_similarity(context_embedding, caption_embedding.unsqueeze(0))
            similarities.append(score.item())
        return similarities

    def embed_text(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding

    def embed_texts(self, texts):
        inputs = self.bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def download_video_from_s3(self, s3_bucket, s3_key):
        response = self.s3.get_object(Bucket=s3_bucket, Key=s3_key)
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

    def convert_image_to_byte_array(self, image):
        byte_arr = BytesIO()
        image.save(byte_arr, format='JPEG')
        byte_arr.seek(0)
        return byte_arr

    def upload_frame(self, s3_bucket, s3_path, frame):
        s3_path_frame = s3_path.split('.')[0] + "/" + str(uuid.uuid4()) + ".jpeg"
        self.s3.upload_fileobj(self.convert_image_to_byte_array(frame), s3_bucket, s3_path_frame)
        return s3_path_frame

class ProcessFramesResource:
    def on_post(self, req, resp):
        data = req.media
        s3_path = data.get('s3_path')
        context = data.get('query')
        if not s3_path or not context:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "s3_path and context are required"}
            return

        retriever = FrameRetriever()
        s3_bucket = s3_path.split('/')[2]
        s3_key = '/'.join(s3_path.split('/')[3:])

        # Download and process the video
        video_path = retriever.download_video_from_s3(s3_bucket, s3_key)
        images = retriever.extract_frames_from_video(video_path)

        # Generate captions for each frame
        frame_captions = retriever.generate_captions(images)

        # Calculate similarity scores between the context and each frame caption
        similarity_scores = retriever.calculate_similarity(context, frame_captions)

        # Select the most similar frame
        max_score_index = np.argmax(similarity_scores)
        most_similar_frame = images[max_score_index]
        most_similar_caption = frame_captions[max_score_index]
        max_score = similarity_scores[max_score_index]

        # Upload the most similar frame and get its S3 path
        s3_frame_path = retriever.upload_frame(s3_bucket, s3_key, most_similar_frame)

        result = {"frame": { "bucket": s3_bucket,"key":s3_frame_path},  "caption": most_similar_caption, "similarity_score": max_score}
        resp.media = result
