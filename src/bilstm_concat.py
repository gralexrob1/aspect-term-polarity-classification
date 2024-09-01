# class BiLSTM_ATSC_Concat(nn.Module):
#     def __init__(self, embedding_matrix, hidden_dim, output_dim):
#         super(BiLSTM_ATSC_Concat, self).__init__()
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.tensor(embedding_matrix, dtype=torch.float32)
#         )
#         self.lstm = nn.LSTM(
#             input_size=300
#             + 300
#             + 300,  # Sentence embedding + Aspect term embedding + Aspect category embedding
#             hidden_size=hidden_dim,
#             bidirectional=True,
#             batch_first=True,
#         )
#         self.attention = Attention(hidden_dim * 2)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, sentence_indices, aspect_term_indices, aspect_category_indices):
#         sentence_embeddings = self.embedding(sentence_indices)
#         aspect_term_embeddings = self.embedding(aspect_term_indices)
#         aspect_category_embeddings = self.embedding(aspect_category_indices)

#         # Average aspect term embeddings and aspect category embeddings
#         aspect_term_embedding = torch.mean(aspect_term_embeddings, dim=1)
#         aspect_category_embedding = torch.mean(aspect_category_embeddings, dim=1)

#         # Expand aspect embeddings to match the sequence length of sentence embeddings
#         aspect_term_embedding = aspect_term_embedding.unsqueeze(1).repeat(
#             1, sentence_embeddings.size(1), 1
#         )
#         aspect_category_embedding = aspect_category_embedding.unsqueeze(1).repeat(
#             1, sentence_embeddings.size(1), 1
#         )

#         # Concatenate sentence, aspect term, and aspect category embeddings
#         combined_embeddings = torch.cat(
#             (sentence_embeddings, aspect_term_embedding, aspect_category_embedding), dim=-1
#         )

#         # Pass through LSTM
#         lstm_out, _ = self.lstm(combined_embeddings)

#         # Apply attention
#         context_vector = self.attention(lstm_out)

#         # Final sentiment prediction
#         output = self.fc(context_vector)
#         return output
