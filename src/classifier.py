from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from atae_lstm import ATAE_LSTM_Dataset, ATAE_LSTM_Model
from bilstm_attention import BiLSTM_Attention_Dataset, BiLSTM_Attention_Model
from glove_embedding import setup_glove
from preprocess import load_glove_embeddings
from tdlstm import TD_LSTM_Dataset, TD_LSTM_Model


class Classifier:
    def __init__(self):
        # Embedding
        self.glove_file_path = "glove.6B.300d.txt"
        self.max_len = 50

        setup_glove(self.glove_file_path)
        self.word_to_idx, embedding_matrix = load_glove_embeddings(self.glove_file_path)

        # Dataset
        # self.dataset = TD_LSTM_Dataset
        self.dataset = BiLSTM_Attention_Dataset
        # self.dataset = ATAE_LSTM_Dataset
        self.batch_size = 64

        # Model
        # model = TD_LSTM_Model
        model = BiLSTM_Attention_Model
        # model = ATAE_LSTM_Model
        self.hidden_dim = 128
        self.output_dim = 3
        self.bidirectional = True
        self.num_layers = 1
        self.dropout_rate = 0.7

        self.model = model(
            embedding_matrix,
            self.hidden_dim,
            self.output_dim,
            self.bidirectional,
            self.num_layers,
            self.dropout_rate,
            # self.max_len,  # when ATAE LSTM Model
        )

        if isinstance(self.model, TD_LSTM_Model):
            print("Running TD LSTM model.")
        elif isinstance(self.model, BiLSTM_Attention_Model):
            print("Running BiLSTM with Attention model.")
        elif isinstance(self.model, ATAE_LSTM_Model):
            print("Running ATAE LSTM Model.")
        else:
            print("Unrecognized model.")

        # Optimizer
        self.optimizer = optim.AdamW
        self.lr = 1e-3
        self.weight_decay = 1e-4
        # self.momentum = 0.9

        # self.scheduler_params = {"mode": "min", "factor": 0.1, "patience": 3, "verbose": True}

        # Learning
        self.num_epochs = 12

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        train_dataset = self.dataset(train_filename, self.word_to_idx, self.max_len)
        dev_dataset = self.dataset(dev_filename, self.word_to_idx, self.max_len)

        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(device)

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            # momentum = self.momentum
        )

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            all_preds = []
            all_labels = []

            for batch in train_data_loader:

                if isinstance(self.model, TD_LSTM_Model):
                    left_sentence_indices = batch["left_sentence_indices"].to(device)
                    right_sentence_indices = batch["right_sentence_indices"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    outputs = self.model(left_sentence_indices, right_sentence_indices)

                if isinstance(self.model, BiLSTM_Attention_Model):
                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    outputs = self.model(sentence_indices, aspect_indices)

                if isinstance(self.model, ATAE_LSTM_Model):
                    pad_index = self.word_to_idx["<PAD>"]

                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)
                    labels = batch["labels"].to(device)

                    bs = sentence_indices.size(0)

                    h0, c0 = self.model.init_prev_hidden(bs)
                    h0 = h0.to(device)
                    c0 = c0.to(device)

                    optimizer.zero_grad()
                    outputs, H = self.model(sentence_indices, aspect_indices, pad_index, (h0, c0))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Accuracy: {accuracy}")

            val_loss = self.evaluate(dev_data_loader, device)
            # scheduler.step(val_loss)

    def evaluate(self, data_loader, device):
        self.model.eval()
        all_preds = []
        all_labels = []

        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():

            for batch in data_loader:

                if isinstance(self.model, TD_LSTM_Model):
                    left_sentence_indices = batch["left_sentence_indices"].to(device)
                    right_sentence_indices = batch["right_sentence_indices"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = self.model(left_sentence_indices, right_sentence_indices)

                if isinstance(self.model, BiLSTM_Attention_Model):
                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = self.model(sentence_indices, aspect_indices)

                if isinstance(self.model, ATAE_LSTM_Model):
                    pad_index = self.word_to_idx["<PAD>"]

                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)
                    labels = batch["labels"].to(device)

                    bs = sentence_indices.size(0)

                    h0, c0 = self.model.init_prev_hidden(bs)
                    h0 = h0.to(device)
                    c0 = c0.to(device)

                    outputs, H = self.model(sentence_indices, aspect_indices, pad_index, (h0, c0))

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Eval Accuracy: {accuracy}")

        avg_val_loss = total_loss / len(data_loader)
        return avg_val_loss

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        dataset = self.dataset(data_filename, self.word_to_idx, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.to(device)
        self.model.eval()
        all_preds = []

        with torch.no_grad():

            for batch in dataloader:

                if isinstance(self.model, TD_LSTM_Model):
                    left_sentence_indices = batch["left_sentence_indices"].to(device)
                    right_sentence_indices = batch["right_sentence_indices"].to(device)

                    outputs = self.model(left_sentence_indices, right_sentence_indices)

                if isinstance(self.model, BiLSTM_Attention_Model):
                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)

                    outputs = self.model(sentence_indices, aspect_indices)

                if isinstance(self.model, ATAE_LSTM_Model):
                    pad_index = self.word_to_idx["<PAD>"]

                    sentence_indices = batch["sentence_indices"].to(device)
                    aspect_indices = batch["aspect_indices"].to(device)

                    bs = sentence_indices.size(0)

                    h0, c0 = self.model.init_prev_hidden(bs)
                    h0 = h0.to(device)
                    c0 = c0.to(device)

                    outputs, H = self.model(sentence_indices, aspect_indices, pad_index, (h0, c0))

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        polarity_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
        return [polarity_mapping[p] for p in all_preds]
