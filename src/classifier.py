import re
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


class ATSC_Dataset(Dataset):
    def __init__(self, filepath, word_to_idx, max_len):
        self.data = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["polarity", "aspect_category", "aspect_term", "offset", "sentence"],
        )
        self.data["polarity"] = self.data["polarity"].map(
            {"negative": 0, "neutral": 1, "positive": 2}
        )
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        aspect_term = row["aspect_term"]
        sentence = row["sentence"]
        polarity = row["polarity"]

        aspect_term_tokens = self.simple_tokenize(aspect_term.lower())
        sentence_tokens = self.simple_tokenize(sentence.lower())

        aspect_term_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in aspect_term_tokens
        ]
        sentence_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in sentence_tokens
        ]

        aspect_term_indices = self.pad_sequence(aspect_term_indices, self.max_len)
        sentence_indices = self.pad_sequence(sentence_indices, self.max_len)

        return {
            "aspect_term_indices": torch.tensor(aspect_term_indices, dtype=torch.long),
            "sentence_indices": torch.tensor(sentence_indices, dtype=torch.long),
            "labels": torch.tensor(polarity, dtype=torch.long),
        }

    def simple_tokenize(self, text):
        return re.findall(r"\b\w+\b", text)

    def pad_sequence(self, seq, max_len):
        if len(seq) < max_len:
            seq += [self.word_to_idx["<PAD>"]] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        attn_scores = self.attention_weights(hidden_states).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)
        return context_vector


class BiLSTM_ATSC(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(BiLSTM_ATSC, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.lstm = nn.LSTM(
            input_size=300, hidden_size=hidden_dim, bidirectional=True, batch_first=True
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sentence_indices, aspect_term_indices):
        sentence_embeddings = self.embedding(sentence_indices)
        lstm_out, _ = self.lstm(sentence_embeddings)
        context_vector = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output


class Classifier:
    def __init__(self):
        self.glove_file_path = "glove.6B.300d.txt"
        self.num_epochs = 10
        self.lr = 1e-3
        self.max_len = 50
        self.hidden_dim = 128
        self.output_dim = 3
        self.batch_size = 32
        self.word_to_idx = None
        self.model = None

    def load_glove_embeddings(self, glove_file_path):
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        embeddings = []
        embeddings.append(np.zeros(300))  # Padding token
        embeddings.append(np.random.normal(0, 1, 300))  # Unknown token

        with open(glove_file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                word_to_idx[word] = len(word_to_idx)
                embeddings.append(vector)

        embeddings = np.array(embeddings)
        return word_to_idx, embeddings

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        glove_file_path = self.glove_file_path
        self.word_to_idx, embedding_matrix = self.load_glove_embeddings(glove_file_path)

        train_dataset = ATSC_Dataset(train_filename, self.word_to_idx, self.max_len)
        dev_dataset = ATSC_Dataset(dev_filename, self.word_to_idx, self.max_len)

        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        dev_data_loader = DataLoader(dev_dataset, batch_size=self.batch_size)

        self.model = BiLSTM_ATSC(embedding_matrix, self.hidden_dim, self.output_dim)
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in train_data_loader:
                sentence_indices = batch["sentence_indices"].to(device)
                aspect_term_indices = batch["aspect_term_indices"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = self.model(sentence_indices, aspect_term_indices)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.evaluate(dev_data_loader, device)

    def evaluate(self, data_loader, device):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in data_loader:
                sentence_indices = batch["sentence_indices"].to(device)
                aspect_term_indices = batch["aspect_term_indices"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(sentence_indices, aspect_term_indices)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy}")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        dataset = ATSC_Dataset(data_filename, self.word_to_idx, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.to(device)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                sentence_indices = batch["sentence_indices"].to(device)
                aspect_term_indices = batch["aspect_term_indices"].to(device)

                outputs = self.model(sentence_indices, aspect_term_indices)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())

        polarity_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        return [polarity_mapping[p] for p in all_preds]
