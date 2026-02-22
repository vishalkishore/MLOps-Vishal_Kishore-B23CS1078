import gzip
import json
import os
import random

import numpy as np
import requests
import torch

from utils import GENRE_URL_DICT


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    print(f"  HTTP {response.status_code}")
    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(genre_url_dict=None, head=10000, sample_size=2000):
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT

    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=head, sample_size=sample_size)
    return genre_reviews


def split_data(genre_reviews, train_per_genre=800, total_per_genre=1000):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        sampled = random.sample(reviews, min(total_per_genre, len(reviews)))
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_data(tokenizer, texts, labels, label2id, max_length=512):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    labels_encoded = [label2id[y] for y in labels]
    return ReviewDataset(encodings, labels_encoded)
