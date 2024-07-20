import os
import uuid
import boto3
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import falcon
from moviepy.editor import VideoFileClip
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer, BertModel
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from TrainNearestFrameToContextForDomainSpecificTask import LoRAImageTextRetrievalModel

# Set the environment variable to avoid OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

class FrameRetriever:
    def __init__(self):
         # AWS credentials
        self.aws_access_key_id = 'key'
        self.aws_secret_access_key = 'secret'
        self.aws_region = 'us-east-1'

        # S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )

        # Determine device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(datetime.now(),f"Using device: {self.device}")

        # Load BLIP model and processor
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
        # Load BERT tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Load the LoRA model
        self.lora_model = LoRAImageTextRetrievalModel().to(self.device)
        self.lora_model.load_state_dict(torch.load('./models/lora_image_text_retrieval_model.pth'))
        self.lora_model.eval()

        # Transform for converting PIL images to tensors
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])

    def generate_captions(self, images, batch_size, num_workers):
        dataset = ImageDataset(images, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        total_batches = len(images) // batch_size + 1
        batch_running = 1
        captions = []
        for batch in dataloader:
            inputs = self.blip_processor(images=batch, return_tensors="pt", padding=True, do_rescale=False).to(self.device)
            out = self.blip_model.generate(**inputs)
            batch_captions = [self.blip_processor.decode(o, skip_special_tokens=True) for o in out]
            captions.extend(batch_captions)
            print(datetime.now(), "[", batch_running, "/", total_batches, "] captions generated for one batch of frames")
            batch_running += 1
        return captions

    def calculate_similarity(self, context, images, batch_size, num_workers):
        dataset = ImageDataset(images, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        
        context_inputs = self.bert_tokenizer(context, return_tensors='pt', padding=True, truncation=True).to(self.device)
        similarities = []
        with torch.no_grad():
            for images_batch in dataloader:
                images_batch = images_batch.to(self.device)
                image_features, text_features = self.lora_model(images_batch, context_inputs.input_ids, context_inputs.attention_mask)
                batch_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                similarities.extend(batch_similarity.cpu().numpy())
        return similarities

    def download_video_from_s3(self, s3_bucket, s3_key):
        response = self.s3.get_object(Bucket=s3_bucket, Key=s3_key)
        video_data = response['Body'].read()
        video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_data)
        print(datetime.now(),"download of video completed")
        return video_path

    def extract_frames_from_video(self, video_path, fps):
        clip = VideoFileClip(video_path, fps_source='fps', audio=False, target_resolution=(224, 224), resize_algorithm='fast_bilinear')
        frames = []
        for frame in clip.iter_frames(fps=fps):
            img = Image.fromarray(frame)
            frames.append(img)
        print(datetime.now(),"frames extracted, total size :", len(frames))
        return frames

    def del_video_after_frames_retrival(self, video_path):
        if os.path.exists(video_path):
            os.remove(video_path)
        print(datetime.now(),"video deleted after extracting the frames")
        return

    def convert_image_to_byte_array(self, image):
        byte_arr = BytesIO()
        image.save(byte_arr, format='JPEG')
        byte_arr.seek(0)
        return byte_arr

    def upload_frame(self, s3_bucket, s3_path, frame):
        s3_path_frame = s3_path.split('.')[0] + "/" + str(uuid.uuid4()) + ".jpeg"
        self.s3.upload_fileobj(self.convert_image_to_byte_array(frame), s3_bucket, s3_path_frame)
        print(datetime.now(),"upload of nearest frame completd")
        return s3_path_frame

class ProcessFramesResource:
    def on_post(self, req, resp):
        print(datetime.now(),"recieved request to search nearest frame")
        data = req.media
        s3_path = data.get('s3Path')
        context = data.get('query')
        if not s3_path or not context:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "s3_path and context are required"}
            return

        retriever = FrameRetriever()
        s3_bucket = s3_path.split('/')[2]
        s3_key = '/'.join(s3_path.split('/')[3:])

        # Download and process the video, and delete the raw file after retrieving the frames
        video_path = retriever.download_video_from_s3(s3_bucket, s3_key)
        images = retriever.extract_frames_from_video(video_path, fps=30)
        retriever.del_video_after_frames_retrival(video_path)

        # Generate captions for each frame
        frame_captions = retriever.generate_captions(images, batch_size=32, num_workers=8)

        # Calculate similarity scores between the context and each frame
        similarity_scores = retriever.calculate_similarity(context, images, batch_size=32, num_workers=8)

        # Select the most similar frame
        max_score_index = np.argmax(similarity_scores)
        most_similar_frame = images[max_score_index]
        most_similar_caption = frame_captions[max_score_index]
        max_score = similarity_scores[max_score_index]

        # Upload the most similar frame and get its S3 path
        s3_frame_path = retriever.upload_frame(s3_bucket, s3_key, most_similar_frame)

        result = {"frame": { "bucket": s3_bucket, "key": s3_frame_path}, "caption": most_similar_caption, "similarityScore": max_score}
        resp.media = result