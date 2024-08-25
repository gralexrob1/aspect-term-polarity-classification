from typing import List

import torch
from torch.utils.data import Dataset


class AspectBasedSentimentDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.data = self._load_data(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding["labels"] = torch.tensor(item["label"])
        return encoding

    def _load_data(self, filename):
        data = []
        with open(filename, "r") as file:
            for line in file:
                fields = line.strip().split("\t")
                label = {"positive": 2, "neutral": 1, "negative": 0}[fields[0]]
                aspect_category = fields[1]
                aspect_category = aspect_category.split("#")[1]
                target_term = fields[2]
                sentence = fields[4]
                text = (
                    f"What do you think about the {target_term} {aspect_category}? [SEP] {sentence}"
                )
                data.append({"text": text, "label": label})
        return data


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
    """

    ############################################# complete the classifier class below

    def __init__():
        """
        This should create and initilize the model. Does not take any arguments.

        """

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
