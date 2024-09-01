# class TC_LSTM(nn.Module):
#     def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers, dropout_rate):
#         super(TC_LSTM, self).__init__()
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.tensor(embedding_matrix, dtype=torch.float32)
#         )

#         self.lstm = nn.LSTM(
#             input_size=300 + 300,
#             hidden_size=hidden_dim,
#             batch_first=True,
#             num_layers=num_layers,
#             dropout=dropout_rate if num_layers > 1 else 0,
#         )

#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, sentence_indices, aspect_term_indices):
#         sentence_embeddings = self.embedding(sentence_indices)
#         # [batch_size, seq_len, embedding_dim]

#         aspect_term_embeddings = self.embedding(aspect_term_indices)
#         # [batch_size, seq_len, embedding_dim]
#         aspect_term_embeddings = torch.mean(aspect_term_embeddings, dim=1)
#         # [batch_size, embedding_dim]
#         aspect_term_embeddings = aspect_term_embeddings.unsqueeze(1).repeat(
#             1, sentence_embeddings.size(1), 1
#         )
#         # [batch_size, seq_len, embedding_dim]

#         combined_embeddings = torch.cat((sentence_embeddings, aspect_term_embeddings), dim=2)
#         # [batch_size, seq_len, embedding_dim*2]

#         lstm_out, _ = self.lstm(combined_embeddings)
#         # [batch_size, seq_len, hidden_dim]

#         lstm_out = lstm_out[:, -1, :]
#         # [batch_size, hidden_dim]
#         output = self.fc(lstm_out)
#         # [batch_size, output_dim]

#         return output
