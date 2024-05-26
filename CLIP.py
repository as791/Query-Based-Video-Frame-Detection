import torch
import torch.nn.functional as F
from transformers import ViTModel, ViTFeatureExtractor,BertModel,BertTokenizerFast
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
import numpy as np
import requests



class CLIPModel(nn.Module):
    def __init__(self, d_e, device):
        super(CLIPModel, self).__init__()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224',trust_remote_code=True).to(device)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',trust_remote_code=True).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',trust_remote_code=True)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224',trust_remote_code=True)

        # Learned projections
        d_i = self.image_encoder.config.hidden_size
        d_t = self.text_encoder.config.hidden_size
        self.W_i = nn.Linear(d_i, d_e, bias=False).to(device)
        self.W_t = nn.Linear(d_t, d_e, bias=False).to(device)

        # Learned temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * 0.07).to(device)

    def forward(self, images, texts, device):
        # Encode images
        images = self.feature_extractor(images=images, return_tensors="pt").to(device).pixel_values
        I_f = self.image_encoder(images).pooler_output

        # Encode texts
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        T_f = self.text_encoder(**tokens).pooler_output

        # Project to joint embedding space
        I_e = F.normalize(self.W_i(I_f), dim=1)
        T_e = F.normalize(self.W_t(T_f), dim=1)

        # Scaled pairwise cosine similarities
        logits = torch.matmul(I_e, T_e.T) * torch.exp(self.temperature)

        return logits

def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def compute_loss(logits, n):
    labels = torch.arange(n, device=logits.device)
    loss_i = cross_entropy_loss(logits, labels)
    loss_t = cross_entropy_loss(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    return loss

def train_clip_model(model, train_loader, optimizer,device,label_mapper):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        images = data[0]
        # Generate dummy texts for simplicity; replace with actual captions in practice
        texts = ["A photo of a {label}".format(label=label_mapper[label]) for label in data[1]]

        logits = model(images, texts,device)
        loss = compute_loss(logits,len(images))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print("one batch completed!!")

    print("one epoch completed!!")

    return total_loss / len(train_loader)

def validate_clip_model(model, val_loader,device,label_mapper):
    model.eval()
    total_loss = 0
    with torch.no_grad():
         for i, data in enumerate(val_loader):
            images = data[0]
            texts = ["A photo of a {label}".format(label=label_mapper[label]) for label in data[1]]
            
            logits = model(images, texts,device)
            loss = compute_loss(logits, len(images))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        image = item['image']
        if len(np.array(image).shape)==2 or np.array(image).shape[2] == 1:
            continue
        images.append(item['image'])
        labels.append(item['label'])
    return images, labels

def main():
    # Load the label mapping file
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    response = requests.get(url)
    labels = response.text.split('\n')
    label_mapper = {i: label for i, label in enumerate(labels) if label}

    # Load the ImageNet-1K dataset
    train_dataset = load_dataset('imagenet-1k', split='train', streaming=True, trust_remote_code=True)
    validation_dataset = load_dataset('imagenet-1k', split='validation', streaming=True,trust_remote_code=True)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=128, collate_fn=collate_fn)
    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = CLIPModel(d_e=256,device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train_clip_model(model, train_loader, optimizer,device,label_mapper)
        val_loss = validate_clip_model(model, val_loader, device,label_mapper)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()

