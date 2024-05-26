from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import faiss
import torch
import numpy as np
import pickle

def index_corpus():
    # Load BERT model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # # Load the OpenWebText dataset
    # dataset = load_dataset("openwebtext", split="train[:1%]")  # Use a subset for demonstration
    # corpus = dataset['text']  # Extract the text data

    # Load the MS COCO dataset
    dataset = load_dataset("lmms-lab/COCO-Caption2017",split='val')  # Use a subset for demonstration
    # print(dataset)
    corpus = [item for items in dataset['answer'] for item in items]  # Extract the captions


    embeddings = []
    for text in corpus:
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        embedding = bert_model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and corpus
    faiss.write_index(index, "faiss_index.bin")
    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

    print("Indexing complete")

if __name__ == "__main__":
    index_corpus()
