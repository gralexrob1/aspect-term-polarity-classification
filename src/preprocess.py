import re

import numpy as np
import pandas as pd


def load_data(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["polarity", "aspect_category", "aspect_term", "offset", "sentence"],
    )
    return df


def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.replace("_", " "))


def remove_stopwords(text, stopwords):
    if isinstance(text, str):
        text = " ".join(word for word in text.split() if word not in stopwords)
    if isinstance(text, list):
        text = [word for word in text if word not in stopwords]
    return text


def pad_sequence(seq, word_to_idx, max_len, pad_on_left=False):
    if len(seq) < max_len:
        padding = [word_to_idx["<PAD>"]] * (max_len - len(seq))
        if pad_on_left:
            seq = padding + seq
        else:
            seq = seq + padding
    else:
        if pad_on_left:
            seq = seq[-max_len:]
        else:
            seq = seq[:max_len]
    return seq


def load_glove_embeddings(glove_file_path):
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


def extract_offset(text):
    start_index, end_index = text.split(":")
    return int(start_index), int(end_index)
