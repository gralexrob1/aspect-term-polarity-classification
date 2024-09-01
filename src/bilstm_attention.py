import torch
import torch.nn as nn
from torch.utils.data import Dataset

from preprocess import load_data, pad_sequence, remove_stopwords, simple_tokenize
from stopwords import STOPWORDS


class BiLSTM_Attention_Dataset(Dataset):
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
        aspect_category = row["aspect_category"]
        aspect_term = row["aspect_term"]
        sentence = row["sentence"]

        # Tokenization
        aspect_category_tokens = simple_tokenize(aspect_category.lower())
        aspect_term_tokens = simple_tokenize(aspect_term.lower())
        sentence_tokens = simple_tokenize(sentence.lower())

        # Stopwords
        aspect_term_tokens = remove_stopwords(aspect_term_tokens, STOPWORDS)
        sentence_tokens = remove_stopwords(sentence_tokens, STOPWORDS)

        aspect_tokens = aspect_category_tokens + aspect_term_tokens

        # Indexation
        aspect_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in aspect_tokens
        ]
        sentence_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in sentence_tokens
        ]

        # Padding
        aspect_indices = pad_sequence(aspect_indices, self.word_to_idx, self.max_len)
        sentence_indices = pad_sequence(sentence_indices, self.word_to_idx, self.max_len)

        return {
            "aspect_indices": torch.tensor(aspect_indices, dtype=torch.long),
            "sentence_indices": torch.tensor(sentence_indices, dtype=torch.long),
            "labels": torch.tensor(polarity, dtype=torch.long),
        }


# class Attention(nn.Module): # 78.72
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

#     def forward(self, hidden_states):
#         scores = self.attention_weights(hidden_states).squeeze(-1)
#         weights = torch.softmax(scores, dim=1)
#         context = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)
#         return context, weights


# class Attention(nn.Module): # Unefficient
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

#     def forward(self, lstm_output):
#         scores = torch.matmul(lstm_output, self.attention_vector)
#         weights = torch.softmax(scores, dim=1)
#         context = torch.sum(weights.unsqueeze(2) * lstm_output, dim=1)
#         return context, weights


class Attention(nn.Module):  # 79.20
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.va = nn.Parameter(torch.randn(hidden_size))

    def forward(self, lstm_output):
        scores = torch.tanh(self.Wa(lstm_output) + self.Ua(lstm_output))
        scores = torch.matmul(scores, self.va)
        weights = torch.nn.functional.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(2) * lstm_output, dim=1)
        return context, weights


class BiLSTM_Attention_Model(nn.Module):
    def __init__(
        self, embedding_matrix, hidden_dim, output_dim, bidirectional, num_layers, dropout_rate
    ):
        super(BiLSTM_Attention_Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sentence_indices, aspect_indices):
        sentence_embeddings = self.embedding(sentence_indices)
        aspect_embeddings = self.embedding(aspect_indices)

        sentence_lstm_out, _ = self.lstm(sentence_embeddings)
        aspect_lstm_out, _ = self.lstm(aspect_embeddings)

        sentence_context_vector, _ = self.attention(sentence_lstm_out)
        aspect_context_vector, _ = self.attention(aspect_lstm_out)

        context_vector = torch.cat([sentence_context_vector, aspect_context_vector], dim=1)
        context_vector = self.dropout(context_vector)

        output = self.fc(context_vector)
        return output
