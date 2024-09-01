import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from preprocess import load_data, pad_sequence, remove_stopwords, simple_tokenize
from stopwords import STOPWORDS


class ATAE_LSTM_Dataset(Dataset):
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
        aspect_term = row["aspect_term"]
        sentence = row["sentence"]

        # Tokenization
        aspect_term_tokens = simple_tokenize(aspect_term.lower())
        sentence_tokens = simple_tokenize(sentence.lower())

        # Stopwords
        aspect_term_tokens = remove_stopwords(aspect_term_tokens, STOPWORDS)
        sentence_tokens = remove_stopwords(sentence_tokens, STOPWORDS)

        # Indexation
        aspect_indices = [
            self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in aspect_term_tokens
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


class ATAE_LSTM_Model(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_dim,
        output_dim,
        bidirectional,
        num_layers,
        dropout_rate,
        max_len,
    ):
        super(ATAE_LSTM_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # W_HV * K
        self.w_hv = nn.Linear(
            hidden_dim + self.embedding.embedding_dim, hidden_dim + self.embedding.embedding_dim
        )

        # M=tanh(K)
        # W_M * M
        self.w_m = nn.Linear(hidden_dim + self.embedding.embedding_dim, hidden_dim)

        # alpha = Softmax(W_M * M)
        # r = H alpha
        # W_R * r
        self.w_r = nn.Linear(max_len, hidden_dim)

        # W_HN * h_n
        self.w_hn = nn.Linear(hidden_dim, hidden_dim)

        # W * h*
        self.w_h_star = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sentence_indices, aspect_indices, pad_index, prev_hidden):
        # Sentence embeddings
        # (batch_size, max_len, embedding_dim)
        sentence_embeddings = self.embedding(sentence_indices)

        # Aspect embeddings
        # (batch_size, max_len, embedding_dim)
        aspect_embeddings = self.embedding(aspect_indices)

        # Remove padding elements from aspect embeddings
        pad_mask = (aspect_indices != pad_index).float()
        masked_aspect_embeddings = aspect_embeddings * pad_mask.unsqueeze(-1)

        # Average aspect embeddings
        # (batch_size, embedding_dim)
        # mean_aspect_embeddings = torch.mean(aspect_embeddings, 1)
        sum_aspect_embeddings = masked_aspect_embeddings.sum(dim=1)
        num_non_padding_tokens = pad_mask.sum(dim=1)
        mean_aspect_embeddings = sum_aspect_embeddings / num_non_padding_tokens.unsqueeze(-1)

        # Repeat aspect embeddings
        # (batch_size, max_len, embedding_dim)
        repeated_aspect_embeddings = mean_aspect_embeddings.unsqueeze(1).repeat(
            1, sentence_embeddings.size(1), 1
        )

        # Concatenante sentence embeddings and aspect embeddings
        # (batch_size, max_len, embedding_dim * 2)
        x = torch.cat((sentence_embeddings, repeated_aspect_embeddings), 2)

        H, context = self.lstm(x, prev_hidden)
        H = self.dropout(H)
        h_n = H[:, -1]

        K = self.__get_K(H, x)
        W_H_A = self.w_hv(K)
        M = torch.tanh(W_H_A)
        W_M = self.w_m(M)
        alpha = self.__softmax(W_M, dim=1)

        r = torch.matmul(H, torch.transpose(alpha, 1, 2))

        W_R = self.w_r(r)
        W_HN = self.w_hn(h_n)

        sum = torch.clone(W_R)
        for i, a_batch in enumerate(sum):
            sum[i] = W_R[i] + W_HN[i]

        h_star = torch.tanh(self.dropout(sum))
        W_H_STAR = self.w_h_star(h_star)

        y = self.__softmax(W_H_STAR, dim=1)
        y = y[:, -1]

        return y, H

    def __get_aspect(self, x):
        aspect = x[:, 0, self.embedding.embedding_dim :]
        return aspect

    def __get_K(self, H, x):
        aspect = self.__get_aspect(x)
        K = torch.clone(H)

        K_list = []
        for i, sentence in enumerate(K):
            aspect_tensor = torch.stack([aspect[i] for _ in range(sentence.shape[0])])
            K_list.append(torch.cat((sentence, aspect_tensor), dim=1))
        K = torch.stack(K_list)
        return K

    def __softmax(self, M, dim):
        softmax_list = []
        for i, m in enumerate(M):
            softmax = F.softmax(m, dim=dim)
            softmax_list.append(softmax)
        softmax_tensor = torch.stack(softmax_list)
        return softmax_tensor

    def init_prev_hidden(self, batch_size):
        num_directions = 2 if self.bidirectional else 1

        return (
            torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim),
        )


# class ATAE_LSTM_Model(nn.Module):
#     def __init__(
#         self,
#         embedding_matrix,
#         hidden_dim,
#         output_dim,
#         bidirectional,
#         num_layers,
#         dropout_rate,
#         max_len,
#     ):
#         super(ATAE_LSTM_Model, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.bidirectional = bidirectional
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding.from_pretrained(
#             torch.tensor(embedding_matrix, dtype=torch.float32)
#         )

#         self.lstm = nn.LSTM(
#             input_size=self.embedding.embedding_dim * 2,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout_rate if num_layers > 1 else 0,
#             bidirectional=bidirectional,
#         )

#         self.attention_transform = nn.Linear(
#             hidden_dim + self.embedding.embedding_dim, hidden_dim + self.embedding.embedding_dim
#         )
#         self.attention_weight = nn.Linear(hidden_dim + self.embedding.embedding_dim, hidden_dim)
#         self.context_transform = nn.Linear(max_len, hidden_dim)
#         self.hidden_state_transform = nn.Linear(hidden_dim, hidden_dim)
#         self.final_output_layer = nn.Linear(hidden_dim, output_dim)

#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, sentence_indices, aspect_indices, pad_index, prev_hidden):
#         sentence_embeddings = self.embedding(sentence_indices)
#         aspect_embeddings = self.embedding(aspect_indices)

#         # Create mask for padding
#         pad_mask = (aspect_indices != pad_index).float()
#         masked_aspect_embeddings = aspect_embeddings * pad_mask.unsqueeze(-1)

#         # Compute mean aspect embeddings
#         sum_aspect_embeddings = masked_aspect_embeddings.sum(dim=1)
#         num_non_padding_tokens = pad_mask.sum(dim=1)
#         mean_aspect_embeddings = sum_aspect_embeddings / num_non_padding_tokens.unsqueeze(-1)

#         # Repeat aspect embeddings to match sentence length
#         repeated_aspect_embeddings = mean_aspect_embeddings.unsqueeze(1).repeat(
#             1, sentence_embeddings.size(1), 1
#         )

#         # Concatenate sentence and aspect embeddings
#         combined_embeddings = torch.cat((sentence_embeddings, repeated_aspect_embeddings), dim=2)

#         # Pass through LSTM
#         lstm_output, hidden_state = self.lstm(combined_embeddings, prev_hidden)
#         lstm_output = self.dropout(lstm_output)
#         last_hidden_state = lstm_output[:, -1]

#         # Compute attention scores
#         attention_input = self.attention_transform(lstm_output)  # Apply linear transformation
#         attention_input = torch.tanh(attention_input)  # Apply tanh activation
#         attention_scores = self.attention_weight(attention_input)  # Compute attention weights

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention_scores, dim=1)

#         # Compute weighted sum of LSTM outputs
#         weighted_sum = torch.matmul(lstm_output, attention_weights.transpose(1, 2))

#         # Compute context-aware representation
#         context_representation = self.context_transform(weighted_sum) + self.hidden_state_transform(
#             last_hidden_state
#         )
#         context_representation = torch.tanh(self.dropout(context_representation))

#         # Compute final output logits
#         final_output = self.final_output_layer(context_representation)

#         return final_output, lstm_output

#     def init_prev_hidden(self, batch_size):
#         num_directions = 2 if self.bidirectional else 1
#         return (
#             torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim),
#             torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim),
#         )
