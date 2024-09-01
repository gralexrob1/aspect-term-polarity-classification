import torch
import torch.nn as nn
from torch.utils.data import Dataset

from preprocess import (
    extract_offset,
    load_data,
    pad_sequence,
    remove_stopwords,
    simple_tokenize,
)
from stopwords import STOPWORDS


class TD_LSTM_Dataset(Dataset):
    def __init__(self, filepath, word_to_idx, max_len):
        self.data = load_data(filepath)
        self.word_to_idx = word_to_idx
        self.max_len = max_len

        self.data["polarity"] = self.data["polarity"].map(
            {"negative": 0, "neutral": 1, "positive": 2}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        polarity = row["polarity"]
        offset = row["offset"]
        sentence = row["sentence"]

        # Sentence split
        start_index, end_index = extract_offset(offset)
        left_sentence = sentence[:end_index]
        right_sentence = sentence[start_index:]

        # Tokenization
        left_sentence_tokens = simple_tokenize(left_sentence.lower())
        right_sentence_tokens = simple_tokenize(right_sentence.lower())

        # Stopwords
        # left_sentence_tokens = remove_stopwords(left_sentence_tokens, STOPWORDS)
        # right_sentence_tokens = remove_stopwords(right_sentence_tokens, STOPWORDS)

        # Indexation
        left_sentence_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in left_sentence_tokens
        ]
        right_sentence_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"])
            for token in right_sentence_tokens
        ]

        # Padding
        left_sentence_indices = pad_sequence(
            left_sentence_indices, self.word_to_idx, self.max_len, pad_on_left=False
        )
        right_sentence_indices = pad_sequence(
            right_sentence_indices, self.word_to_idx, self.max_len, pad_on_left=True
        )

        return {
            "left_sentence_indices": torch.tensor(left_sentence_indices, dtype=torch.long),
            "right_sentence_indices": torch.tensor(right_sentence_indices, dtype=torch.long),
            "labels": torch.tensor(polarity, dtype=torch.long),
        }


class TD_LSTM_Model(nn.Module):
    def __init__(
        self, embedding_matrix, hidden_dim, output_dim, bidirectional, num_layers, dropout_rate
    ):
        super(TD_LSTM_Model, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )

        self.lstm_left = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.lstm_right = nn.LSTM(
            input_size=300,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        fc_dim = hidden_dim * 2 * 2 if bidirectional else hidden_dim * 2
        self.fc = nn.Linear(fc_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, left_context_indices, right_context_indices):
        left_context_embeddings = self.embedding(left_context_indices)
        right_context_embeddings = self.embedding(right_context_indices)

        # Reverse the right context embeddings to simulate reading right-to-left
        right_context_embeddings = torch.flip(right_context_embeddings, dims=[1])

        left_lstm_out, _ = self.lstm_left(left_context_embeddings)
        right_lstm_out, _ = self.lstm_right(right_context_embeddings)

        left_hidden = left_lstm_out[:, -1, :]
        right_hidden = right_lstm_out[:, -1, :]

        context_vector = torch.cat([left_hidden, right_hidden], dim=1)
        context_vector = self.dropout(context_vector)

        output = self.fc(context_vector)
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
        return output
